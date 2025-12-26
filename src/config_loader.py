import yaml
import os

def load_config(name):    
    config_path = f"../config/{name}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Конфиг {name}.yaml не найден. Искал: {os.path.abspath(config_path)}"
        )
    
    print(f"Загружаю конфиг из: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    return config