from openai import OpenAI
 
if __name__ == '__main__':
    client = OpenAI(
        base_url='https://yunwu.ai/v1',
        api_key='sk-1jlCFDIGEKWqgJVb3hiuOFKz6oNlsjq8WrS9onjrrtn06OP5',
    )
 
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        model="gpt-4.1-nano",
    )
 
    print(chat_completion)