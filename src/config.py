from pathlib import Path
import yaml
"""
编写配置函数，便于后续程序使用配置
"""
#加载config
def load_config(path:str) ->  dict:
    config_path = Path(path)
    
    #假如文件不存在
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found:{config_path}")
    
    #读取配置参数
    with config_path.open('r',encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config

#正常时不返回数据；出错时抛出异常。
def validate_config(path:str) -> None:
    """
    后续编写需要抛出异常的模块
    """
    pass
