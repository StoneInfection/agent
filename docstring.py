import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Настройка модели ---
model = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-oss-20b"))

# --- Системный промпт ---
SYSTEM_PROMPT = """Ты — эксперт по документированию Python-кода.
Твоя задача — сгенерировать docstring для переданной функции в указанном стиле.

Правила:
- Верни ТОЛЬКО строку docstring (включая тройные кавычки).
- Не добавляй никаких объяснений, комментариев или примеров кода вокруг docstring.
- Если переданный код НЕ является функцией — верни строго: ERROR: Переданный код не является функцией Python.
- Поддерживаемые стили: Google, NumPy, reStructuredText."""

# --- Входные данные ---
code = """
def calculate_discount(price: float, discount_percent: float) -> float:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
"""
style = "Google"

# --- Цепочка с шаблоном промпта ---
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Стиль документации: {style}\n\nКод функции:\n```python\n{code}\n```\n\nСгенерируй docstring."),
])

chain = prompt | model | StrOutputParser()

# --- Потоковый вывод ---
print("Потоковый вывод docstring:")
for chunk in chain.stream({"code": code, "style": style}):
    print(chunk, end="", flush=True)
print()

# --- Batch: генерация docstring в разных стилях ---
print("\nBatch — три стиля для одной функции:")
inputs = [
    {"code": code, "style": "Google"},
    {"code": code, "style": "NumPy"},
    {"code": code, "style": "reStructuredText"},
]

responses = chain.batch(inputs)
for inp, response in zip(inputs, responses):
    print(f"\n--- Стиль: {inp['style']} ---")
    print(response)

# --- Граничный случай: код не является функцией ---
print("\n--- Граничный случай: передан класс вместо функции ---")
bad_code = """
class MyClass:
    def __init__(self):
        self.value = 42
"""
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"Стиль документации: Google\n\nКод функции:\n```python\n{bad_code}\n```\n\nСгенерируй docstring."),
]
response = model.invoke(messages)
print(response.content)