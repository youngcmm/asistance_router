from typing_extensions import Literal
from pydantic import BaseModel
# from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from llm import LLM

llm = LLM("/Users/ycm/Library/Mobile Documents/com~apple~CloudDocs/code/multi_class/model")

# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["开药", "知识问答", "相似病案检索"] = Field(
        None, description="The next step in the routing process"
    )
router = llm.with_structured_output(Route)

def llm_call_1(input_: str):
    """开药"""
    result = llm.invoke(input_)
    return result.content


def llm_call_2(input_: str):
    """知识问答"""
    result = llm.invoke(input_)
    return result.content



def llm_call_3(input_: str):
    """相似病案"""
    result = llm.invoke(input_)
    return result.content


def llm_call_router(input_: str):
    """Route the input to the appropriate node"""

    # Augment the LLM with schema for structured output
    router_out = router.invoke(input_)

    print("LLM Router:", router_out)
    return router_out



def router_workflow(input_: str):
    next_step = llm_call_router(input_)
    if next_step == "开药":
        llm_call = llm_call_1
    elif next_step == "知识问答":
        llm_call = llm_call_2
    elif next_step == "相似病案检索":
        llm_call = llm_call_3

    print("LLM Call:", next_step)

    return llm_call(input_).result()

router_workflow("开药")