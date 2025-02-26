from typing import TypedDict, List, Tuple, Annotated, Sequence, Union, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import json
import operator
import requests
from pydantic import Field, model_validator
from agent_tools import hello_world
from IPython.display import Image, display

class OllamaChat(BaseChatModel):
    """Custom chat model for Ollama"""
    
    model_name: str = Field(default="llama2")
    base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.7)
    tools: List[Any] = Field(default_factory=list)
    
    def bind_tools(self, tools: List[Any]) -> None:
        """Bind tools to the chat model."""
        self.tools = tools

    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the environment is properly configured."""
        base_url = values.get("base_url")
        if base_url:
            values["base_url"] = base_url.rstrip("/")
        return values

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "ollama_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the Ollama API."""
        # Add format instructions if we have tools
        if self.tools:
            format_message = SystemMessage(content="""You are a helpful AI assistant with access to tools. When you want to use a tool, you MUST respond in the following format:

```json
{
    "action": "tool_name",
    "action_input": "input to the tool",
    "thought": "your reasoning"
}
```

If you want to respond directly to the human, use this format:
```json
{
    "action": "Final Answer",
    "action_input": "your response to the human",
    "thought": "your reasoning"
}
```

Available tools:
""" + "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools]))
            
            messages = [format_message] + messages
        
        # Prepare the prompt from messages
        prompt = self._convert_messages_to_prompt(messages)
        
        # Prepare the API request
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }
        if stop:
            data["stop"] = stop

        try:
            # Make the API call
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()

            # Extract the generated text
            generated_text = result.get("response", "")

            # Create a ChatGeneration object
            generation = ChatGeneration(
                message=AIMessage(content=generated_text),
                generation_info=dict(finish_reason="stop")
            )

            # Return the ChatResult
            return ChatResult(generations=[generation])

        except Exception as e:
            raise ValueError(f"Error calling Ollama API: {str(e)}")

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a single prompt string."""
        prompt_pieces = []
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_pieces.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_pieces.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_pieces.append(f"Assistant: {message.content}")
            elif isinstance(message, FunctionMessage):
                prompt_pieces.append(f"Function ({message.name}): {message.content}")
        return "\n".join(prompt_pieces)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "base_url": self.base_url
        }

# Define our state
class AgentState(TypedDict):
    """The state of our agent."""
    messages: List[BaseMessage]  # Chat messages

# Create chat model instance
chat = OllamaChat(model_name="llama2")
chat.bind_tools([hello_world])
# Define our nodes
def agent(state: AgentState) -> AgentState:
    """Agent node that processes messages and decides next action."""
    # Get response from chat model
    response = chat.invoke(state["messages"])
    
    # Add response to messages
    state["messages"].append(response)
    
    # Set next to end for now
    state["next"] = "end"
    return state
def route_tools(
    state: AgentState,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END
# Create the graph
def create_graph():
    """Create the graph with basic workflow."""
    # Create graph with state type
    workflow = StateGraph(AgentState)
    
    # Add our nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools=[hello_world]))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
       route_tools,
       {"tools": "tools", END: END}
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge(START, "agent")
    # Compile
    return workflow.compile()

if __name__ == "__main__":
    # Create the graph
    graph = create_graph()
    
    # Create initial state
    state = AgentState(
        messages=[
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="please say hello to Alice using the hello_world tool")
        ],
        next=""
    )
    
    # Run the graph
    result = graph.invoke(state)
    
    # Print results
    print("\nFinal Messages:")
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"\nHuman: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nAssistant: {message.content}")
