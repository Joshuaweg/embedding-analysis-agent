from langchain.tools import tool

@tool
def hello_world(name: str = "World") -> str:
    """Say hello to someone.
    
    Args:
        name: The name of the person to greet. Defaults to "World".
        
    Returns:
        A greeting message.
    """
    return f"Hello, {name}!" 