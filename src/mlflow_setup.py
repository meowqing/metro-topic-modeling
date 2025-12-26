import mlflow

def setup_mlflow(cfg):
    """Простая настройка MLflow"""
    
    # Всегда используем 127.0.0.1
    tracking_uri = "http://127.0.0.1:8080"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✓ Tracking URI: {tracking_uri}")
    
    # Эксперимент
    experiment_name = cfg.get('experiment', {}).get('name', 'topic_modeling')
    mlflow.set_experiment(experiment_name)
    print(f"✓ Experiment: {experiment_name}")
    
    # Принудительно завершаем ВСЕ активные runs
    try:
        while mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
    except:
        pass
    
    print("✓ Готово к запуску нового эксперимента")
    return True