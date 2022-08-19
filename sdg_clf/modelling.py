import onnxruntime


def create_model_for_provider(model_path: str, provider: str = "CPUExecutionProvider"):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session
