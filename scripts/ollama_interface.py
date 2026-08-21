from ollama import Client
import base64
from pathlib import Path
import concurrent.futures
import time
import os  # NEW


OLLAMA_HOST = "localhost:9007"

client = Client(host=OLLAMA_HOST)


def get_ollama_models():
    models_info = client.list()

    models = [model.model for model in models_info.models]
    return models


DEFAULT_OPTIONS = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": -1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.1,
    "stop": [],
    "seed": None,
}


# --- NEW helpers -------------------------------------------------------------

def _is_ctx_or_runner_error(exc: Exception) -> bool:
    """Detect context/runner/load errors that should trigger num_ctx backoff."""
    msg = str(exc).lower()
    return any(
        key in msg
        for key in [
            "status code: 500",
            "model runner has unexpectedly stopped",
            "runner process has terminated",
            "exit status 2",
            "do load request",
            "eof",
        ]
    )


def _get_initial_num_ctx(options: dict):
    """Use options['num_ctx'] or OLLAMA_NUM_CTX if present."""
    if "num_ctx" in options:
        return options["num_ctx"]
    env_ctx = os.environ.get("OLLAMA_NUM_CTX")
    if env_ctx:
        try:
            return int(env_ctx)
        except ValueError:
            pass
    return None


def chat_with_model(model_name, messages, options=None, image_paths=None, stream=False, timeout=20 * 60):
    """
    Wrapper for chatting with an Ollama model, supporting optional image input, streaming, and default options.
    """

    supported_models = get_ollama_models()

    if model_name not in supported_models:
        raise ValueError(
            f"Model '{model_name}' is not supported. Choose from: {supported_models}")

    # Merge default and user-provided options
    final_options = DEFAULT_OPTIONS.copy()
    if options:
        final_options.update(options)

    # NEW: context backoff state
    current_ctx = _get_initial_num_ctx(final_options)
    MIN_CTX = 16_384
    CTX_STEP = 16_384

    images = None
    if image_paths:
        images = []
        for path in image_paths:
            file_path = Path(path)
            if not file_path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            with open(file_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                images.append(encoded)

    final_messages = messages.copy()

    if images:
        temp = final_messages[-1]
        temp["images"] = images
        final_messages[-1] = temp

    def _messages_to_prompt(msgs):
        """Convert chat messages to a plain prompt if we must use generate()."""
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant:")  # Expecting assistant reply next
        return "\n".join(parts)

    # --- Try chat first ---
    try:
        if stream:
            def stream_generator():
                response_stream = client.chat(
                    model=model_name,
                    messages=final_messages,
                    options=final_options,
                    stream=True
                )
                for chunk in response_stream:
                    yield chunk["message"]["content"]
            return stream_generator()
        else:
            for _ in range(5):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        client.chat,
                        model=model_name,
                        messages=final_messages,
                        options=final_options
                    )
                    try:
                        response = future.result(timeout=timeout)
                        return response
                    except concurrent.futures.TimeoutError:
                        print("Timeout reached. Sleeping and retrying...")
                        print("fallback placeholder sleeps for 5 minutes?")
                        time.sleep(60 * 1)
    except Exception as e:
        print(f"[Fallback] chat() failed for {model_name}: {e}")

        # --- NEW: num_ctx backoff for chat errors (non-stream) ---
        if (not stream) and _is_ctx_or_runner_error(e):
            # if we didn't know num_ctx before, start from a big value
            if current_ctx is None:
                # try env again, else fallback to 1M like your server default
                current_ctx = _get_initial_num_ctx(final_options) or 1_048_576
            last_error_str = str(e)

            while current_ctx > MIN_CTX:
                CTX_STEP = current_ctx/8
                current_ctx = max(MIN_CTX, current_ctx - CTX_STEP)
                final_options["num_ctx"] = current_ctx
                print(
                    f"[chat] Detected ctx/runner error. Reducing num_ctx "
                    f"to {current_ctx} and retrying chat..."
                )
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        client.chat,
                        model=model_name,
                        messages=final_messages,
                        options=final_options
                    )
                    try:
                        response = future.result(timeout=timeout)
                        return response
                    except concurrent.futures.TimeoutError:
                        print(
                            "Timeout reached. Sleeping and retrying same num_ctx...")
                        time.sleep(60)
                        continue
                    except Exception as e2:
                        err_str = str(e2)
                        # stop if error changed or no longer a ctx/runner error
                        if (not _is_ctx_or_runner_error(e2)) or (err_str != last_error_str):
                            print(
                                "[chat] Error changed or not ctx/runner related; stopping chat backoff.")
                            e = e2
                            break
                        last_error_str = err_str
                        continue

        print("Using generate() API instead...")

    # --- Fallback to generate() if chat not supported or failed ---
    prompt = _messages_to_prompt(final_messages)
    if stream:
        def stream_generator():
            response_stream = client.generate(
                model=model_name,
                prompt=prompt,
                options=final_options,
                images=images,
                stream=True
            )
            for chunk in response_stream:
                yield chunk.get("response", "")
        return stream_generator()
    else:
        for _ in range(5):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    client.generate,
                    model=model_name,
                    prompt=prompt,
                    options=final_options,
                    images=images
                )
                try:
                    response = future.result(timeout=timeout)
                    # Make return shape consistent with chat
                    return {
                        "model": model_name,
                        "done": True,
                        "message": {"role": "assistant", "content": response.get("response", "")},
                        **response
                    }
                except concurrent.futures.TimeoutError:
                    print("Timeout reached in generate(). Retrying...")
                    time.sleep(60)
                except Exception as e:
                    # --- NEW: num_ctx backoff loop for generate errors ---
                    if _is_ctx_or_runner_error(e):
                        if current_ctx is None:
                            current_ctx = _get_initial_num_ctx(
                                final_options) or 1048576
                        if current_ctx <= MIN_CTX:
                            print(
                                "[generate] num_ctx already at floor; not backing off further.")
                            print(f"generate() failed for {model_name}: {e}")
                            raise
                        last_error_str = str(e)

                        while current_ctx > MIN_CTX:
                            current_ctx = max(MIN_CTX, current_ctx - CTX_STEP)
                            final_options["num_ctx"] = current_ctx
                            print(
                                f"[generate] Detected ctx/runner error. Reducing num_ctx "
                                f"to {current_ctx} and retrying generate..."
                            )
                            with concurrent.futures.ThreadPoolExecutor() as executor2:
                                future2 = executor2.submit(
                                    client.generate,
                                    model=model_name,
                                    prompt=prompt,
                                    options=final_options,
                                    images=images
                                )
                                try:
                                    response2 = future2.result(timeout=timeout)
                                    return {
                                        "model": model_name,
                                        "done": True,
                                        "message": {"role": "assistant",
                                                    "content": response2.get("response", "")},
                                        **response2
                                    }
                                except concurrent.futures.TimeoutError:
                                    print(
                                        "Timeout reached in generate() with backoff. Retrying same num_ctx...")
                                    time.sleep(60)
                                    continue
                                except Exception as e2:
                                    err_str = str(e2)
                                    if (not _is_ctx_or_runner_error(e2)) or (err_str != last_error_str):
                                        print(
                                            "[generate] Error changed or not ctx/runner related; stopping generate backoff.")
                                        print(
                                            f"generate() failed for {model_name}: {e2}")
                                        raise
                                    last_error_str = err_str
                                    continue
                    else:
                        print(f"generate() failed for {model_name}: {e}")
                        raise
