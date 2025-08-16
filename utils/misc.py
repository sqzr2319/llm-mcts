# def basic_response_filter(response:str) -> str:
#     response = response.strip('\n ')
#     response = response.replace("<think>\n</think>", "")
#     return response

def basic_response_filter(response:str) -> str:
    # 通过 '</think>' 分割字符串，并取分割后的最后一部分。
    # 这可以有效地移除 '</think>' 标签以及它之前的所有思考过程文本。
    # maxsplit=1 参数确保我们只在第一个 '</think>' 处分割，这通常是期望的行为。
    # 如果 '</think>' 不存在，split 会返回一个只包含原始字符串的列表，[-1] 仍然能正确工作。
    response = response.split('</think>', 1)[-1]
    
    # 移除结果字符串开头和结尾可能存在的任何多余的换行符或空格。
    response = response.strip('\n ')
    
    return response
