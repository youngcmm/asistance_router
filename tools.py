from langchain_core.tools import tool
from predict import TCMPredictor
@tool
def multiply(first: int, second: int) -> int:
    "将两个整数相乘。"
    return first * second
@tool
def add(first_int: int, second_int: int) -> int:
    "将两个整数相加。"
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "求底数的幂次方。"
    return base**exponent

@tool
def herbs_to_prescription(patient_info: str) -> str:
    "开药"
    predictor = TCMPredictor(
        model_path='best_model.pth',
        herb_file_path='data/herb_count.txt'
    )
    herbs, _ = predictor.predict(patient_info, top_k=15)
    return "、".join(herbs)