import ast
import operator
def calculator_skill(expression):
    expression = expression.replace("CALCULATE:","").strip()
    expression = expression.splitlines()[0].strip()
    operators = {ast.Add: operator.add,ast.Sub: operator.sub,ast.Mult: operator.mul,ast.Div: operator.truediv,ast.Mod: operator.mod,ast.Pow: operator.pow}
    def calculate(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            left = calculate(node.left)
            right = calculate(node.right)
            op = operators[type(node.op)]
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            value = calculate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        raise ValueError("不支持的计算表达式")
    tree = ast.parse(expression,mode="eval")
    return calculate(tree.body)
