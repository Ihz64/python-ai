import re
import math
from typing import Optional, Dict, Any

class MathEngine:
    """
    High-speed 'Rechenprofi' Math & Logic Engine.
    Instantly detects and evaluates mathematical expressions in German and English.
    """

    def __init__(self):
        # Allowed mathematical functions and constants for safe eval
        self.safe_dict = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "factorial": math.factorial
        }

    def extract_and_evaluate(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detects mathematical questions or expressions in input text and evaluates them immediately.
        """
        if not text:
            return None

        text_clean = text.strip()

        # Handle German/English natural language math questions
        # e.g., "Was ist 12 + 15?", "Berechne 5 * 20", "Wie viel ist 100 / 4"
        pattern = r"(?:was ist|berechne|wie viel ist|wie viel macht|rechne|what is|calculate)?\s*([\d\.\,\s\+\-\*\/\^\(\)\%\!\=\sqrt\sin\cos\tan\pi]+)\s*\?*"

        # Check for explicit math operations in query
        if not re.search(r'[\+\-\*\/\^\%]', text_clean) and not any(kw in text_clean.lower() for kw in ["sqrt", "sin", "cos", "tan", "fakultät", "factorial", "wurzel"]):
            return None

        # Clean math string
        expr = text_clean
        # Remove text prefixes
        for prefix in ["was ist", "berechne", "wie viel ist", "wie viel macht", "rechne", "what is", "calculate"]:
            if expr.lower().startswith(prefix):
                expr = expr[len(prefix):]

        expr = expr.replace("?", "").replace(",", ".").replace("^", "**").strip()

        # Simple safety check: only allow digits, arithmetic operators, parentheses, spaces, and safe word functions
        if not re.match(r'^[0-9\.\s\+\-\*\/\%\(\)\*\*\w]+$', expr):
            return None

        try:
            # Evaluate using safe dictionary
            result = eval(expr, {"__builtins__": None}, self.safe_dict)

            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 6)

            response_str = f"Das Ergebnis der Rechnung '{expr.replace('**', '^')}' ist: {result}"
            return {
                "response": response_str,
                "result": result,
                "expression": expr,
                "confidence": 1.0,
                "source": "math_engine"
            }
        except Exception:
            return None
