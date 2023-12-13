import time
import config
import openai
api = config.OPENAI_API_KEY
#  KEY ='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'
openai.api_key = api 

# gets API Key from environment variable OPENAI_API_KEY
# client = openai.OpenAI()

assistant = openai.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview",
)

thread = openai.beta.threads.create()

message = openai.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
)

run = openai.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account.",
)

time.sleep(20)

run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

if run_status.status == "completed":
    messages = openai.beta.threads.messages.list(thread_id= thread.id)

for msg in messages.data:
    role = msg.role
    content = msg.content[0].text.value
    print(f'{role.capitalize()}: \n{content}')



# print("checking assistant status. ")
# while True:
#     run = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

#     if run.status == "completed":
#         print("done!")
#         messages = openai.beta.threads.messages.list(thread_id=thread.id)

#         print("messages: ")
#         for message in messages:
#             assert message.content[0].type == "text"
#             print({"role": message.role, "message": message.content[0].text.value})

#         openai.beta.assistants.delete(assistant.id)

#         break
#     else:
#         print("in progress...")
#         time.sleep(10)