import os
import json
import requests
from datetime import datetime
from openai import OpenAI

# ========== 配置 ==========
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit(
        "未找到 DEEPSEEK_API_KEY 环境变量。\n"
        "请先运行：setx DEEPSEEK_API_KEY \"你的key\"\n"
        "然后重新打开终端。"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

# ========== 系统提示词 ==========
SYSTEM_PROMPT = (
    "你是一个基于 DeepSeek 模型的智能助手，具备天气查询、数学计算和时间查询能力。"
    "回答时请区分「工具返回的数据」和「你自己的推测」：没有数据支撑的内容，"
    "要明确说明这是你的推测。"
)


# ========== 工具实现 ==========
def get_weather(city: str) -> str:
    """查真实天气（Open-Meteo 免费 API，无需 Key）"""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"},
            timeout=10
        ).json()

        if not geo.get("results"):
            return f"找不到城市：{city}"

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code"
            },
            timeout=10
        ).json()

        cur = w["current"]
        code = cur["weather_code"]
        mapping = {
            0: "晴", 1: "晴间多云", 2: "多云", 3: "阴",
            45: "雾", 48: "雾",
            51: "毛毛雨", 53: "小雨", 55: "中雨",
            61: "小雨", 63: "中雨", 65: "大雨",
            71: "小雪", 73: "中雪", 75: "大雪",
            80: "阵雨", 81: "强阵雨", 82: "暴雨",
            95: "雷雨", 96: "雷雨伴冰雹"
        }
        desc = mapping.get(code, f"未知({code})")
        return f"{city}：{desc}，{cur['temperature_2m']}℃"

    except Exception as e:
        return f"天气查询失败：{e}"


def calculate(expression: str) -> str:
    """算数学表达式（清空 __builtins__，防止代码注入）"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败：{e}"


def get_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_FUNCS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_time": get_time,
}


# ========== 工具声明（JSON Schema）==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如：北京"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，只能包含数字和运算符（+ - * /），如：128*37",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


# ========== Agent 主循环 ==========
def run_agent(user_input: str, history: list = None):
    """执行一轮对话，返回 (回答, 更新后的历史)"""
    messages = list(history) if history else [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_input})

    for step in range(5):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content, messages

        for tc in msg.tool_calls:
            func = TOOL_FUNCS.get(tc.function.name)
            if func is None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"不存在的工具：{tc.function.name}"
                })
                continue

            args = json.loads(tc.function.arguments)
            print(f"\n  🔧 调用 {tc.function.name}({args})")
            result = func(**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

    return "超过最大步数限制", messages


# ========== 交互入口 ==========
if __name__ == "__main__":
    print("=" * 50)
    print(" Agent 已启动（输入 exit 退出）")
    print("=" * 50)

    history = []

    while True:
        user_input = input("\n你：").strip()

        if user_input.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break
        if not user_input:
            continue

        answer, history = run_agent(user_input, history)
        print("Agent：", answer)