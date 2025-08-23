DEFAULT_VALUES = {
    # Signma Threshold
    "sigma_threshold": 10,
    # Graph Size
    "graph_size": 36,
    # System State Threshold
    "gauge_green_min": 0,
    "gauge_green_max": 20,
    "gauge_yellow_min": 20,
    "gauge_yellow_max": 50,
    "gauge_red_min": 50,
    "gauge_red_max": 100,
    # Trainig File path
    "training_file_path": "generated/model_TSGuard.pth",
}

COLOR_MAP = {
    "active": "#90ee90",
    "inactive": "red",
    "imputed": "orange"
}