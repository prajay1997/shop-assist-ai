# Import the libraries
import pandas as pd
import os, json, ast
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import streamlit as st

# read the OpenAI API key
#openai.api_key = open("OpenAI_API_Key.txt", "r").read().strip()
#os.environ['OPENAI_API_KEY'] = openai.api_key

openai.api_key = st.secrets["OPENAI_API_KEY"]

df =  pd.read_csv('laptop_data.csv')


def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"

    example_user_dict = {
        'CPU': "High",
        'GPU': "Medium",
        'Display Quality':"High",
        'Portability': "Low",
        'Multitasking': "Medium",
        'Budget': "150000"
    }

    example_user_req = {
        'CPU': "_",
        'GPU': "_",
        'Display Quality': "_",
        'Portability': "_",
        'Multitasking': "_",
        'Budget': "_"
    }

    system_message = f"""
    You are an intelligent laptop gadget expert and your goal is to find the best laptop for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('CPU','GPU','Display Quality','Portability','Multitasking','Budget') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this
    {{'CPU': 'values','GPU': 'values','Display Quality':'values','Portability':'values','Multitasking':'values','Budget':'values'}}

    The values for all keys, except 'Budget', should be 'Low', 'Medium', or 'High' based on the importance of the corresponding keys, as stated by user.
    All the values in the example dictionary are only representative values.
    
    {delimiter}
    Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised:
    - The values for all keys, except 'Budget', should strictly be either 'Low', 'Medium', or 'High' based on the importance of the corresponding keys, as stated by user.
    - The value for 'Budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 25000 INR. If the user says less than that, please mention that there are no laptops in that range. In case, user does not mention any value for Budget, consider Budget as 300000 which means that the user is okay to spend more
    - Do not randomly assign values to any of the keys.
    - The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    Follow the chain-of-thoughts below and only output the final updated python dictionary for the keys as described in {example_user_req}. \n
    
    {delimiter}
    
    Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their primary use for the laptop is unclear. Ask followup questions to understand their needs.
    You are trying to fill the values of all the keys {{'GPU intensity','Display quality','Portability','Multitasking','Processing speed','Budget'}} in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    If the necessary information has been extracted, only then proceed to the next step. \n
    Otherwise, rephrase the question to capture their profile clearly. \n

    {delimiter}   
    Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys.
    Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.   
    {delimiter}

    {delimiter}
    Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions.
    {delimiter}

    {delimiter}
    Here is a sample conversation between the user and assistant:
    User: "Hi, I am an editor."
    Assistant: "Great! As an editor, you likely require a laptop that can handle demanding tasks. Hence, the laptop should have high multitasking capability. You would also need a high end display for better visuals and editing. May I know what kind of work do you primarily focus on? Are you more involved in video editing, photo editing, or both? Understanding the specific type of editing work will help me tailor my recommendations accordingly. Let me know if my understanding is correct until now."
    User: "I primarily work with After Effects."
    Assistant: "Thank you for providing that information. Working with After Effects involves working with graphics, animations, and rendering, which will require high GPU. Do you work with high-resolution media files, such as 4K videos or RAW photos? Understanding your file sizes will help determine the storage capacity and processing power needed."
    User: "Yes, sometimes I work with 4K videos as well."
    Assistant: "Thank you for the information. Processing 4K vidoes will require a good processor and high GPU. I think we have already determined earlier that you need a high GPU. To ensure I have a complete understanding of your needs, I have one more question: Are you frequently on the go and require a laptop that is lightweight and easy to carry, or do you primarily work from a stationary location?"
    User: "Yes, sometimes I travel but do not carry my laptop."
    Assistant:"Could you kindly let me know your budget for the laptop? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 1.5lakh inr"
    Assistant: "{example_user_dict}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements as precisely and accurate as possible so that you can make decision appropriately.
    """
    conversation = [
        {"role": "system", "content": system_message}
    ]
    return conversation


# Define a Chat Completions API call
# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completions(input, json_format=False):
    MODEL = 'gpt-3.5-turbo'

    # JSON Output
    if json_format == True:
        system_message_json_output = """<<. Return output in JSON format to the key output.>>"""
        # Append the input prompt to include JSON response as specified by OpenAI
        input[0]['content'] += system_message_json_output

        # JSON return type specified
        chat_completion_json = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            response_format = {"type": "json_object"},
            seed = 1234
        )
        output = json.loads(chat_completion_json.choices[0].message.content)

    # Non-JSON Output
    else:
        chat_completion = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            seed = 2345
        )
        output = chat_completion.choices[0].message.content

    return output


def iterate_llm_response(funct, debug_response, num=10):
    """
    Calls a specified function repeatedly and prints the results.
    This function is designed to test the consistency of a response from a given function.
    It calls the function multiple times (default is 10) and prints out the iteration count,
    the function's response(s).

    Parameters
    -------------
        funct : function
        The function to be tested. This function should accept a single argument and return the response value(s)

        debug_response : dict
        The input argument to be passed to 'funct' on each call.

        num : int
        The number of times 'funct' will be called. Defaults to 10.

    """
    for i in range (1, num+1):
        print(f"Iteration: {i}")
        response = funct(debug_response)
        print(response)
        print('#' * 50)



def moderation_check(user_input):
    """
    This function will call OpenAI API to perform moderation check on the user's input.
    Any value below or equal to `user_value` passed as argument is marked as 1 else 0

    Parameters
    -------------
        user_input : str
        Any input which is provided by user for querying OpenAI API

    Returns
    -------------
        status : bool
        Returns True, if any violation is observed, else False
    
    """
    response = openai.moderations.create(input=user_input)
    moderation_output = response.results[0].flagged
    if response.results[0].flagged == True:
        return True
    else:
        return False
    

def intent_confirmation_layer(response_assistant):

    delimiter = "####"

    allowed_values = {'Low','Medium','High'}

    prompt = f"""
    You are a senior evaluator who has an eye for detail.The input text will contain a user requirement captured through 6 keys.
    You are provided an input. You need to evaluate if the input text has the following keys:
    {{
    'CPU':'values',
    'GPU':'values',
    'Display Quality':'values',
    'Portability':'values',
    'Multitasking':'values',
    'Budget':'number'
    }}
    The values for the keys should only be from the allowed values: {allowed_values}. Except, 'Budget' key can take only a numerical value which can be in string representation as well.
    Next you need to evaluate if the keys have the the values filled correctly.
    Only output a one-word string in JSON format at the key 'result' - Yes/No.
    Thought 1 - Output a string 'Yes' if the values are correctly filled for all keys, otherwise output 'No'.
    Thought 2 - If the answer is No, mention the reason in the key 'reason'.
    Thought 3 - Think carefully before the answering.
    """

    messages=[
        {"role": "system", "content":prompt },
        {"role": "user", "content":f"""Here is the input: {response_assistant}""" }
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = messages,
        response_format={ "type": "json_object" },
        seed = 1234
    )

    json_output = json.loads(response.choices[0].message.content)

    return json_output


def dictionary_present(response):
    delimiter = "####"

    user_req = {
        'CPU': "High",
        'GPU': "Medium",
        'Display Quality':"High",
        'Portability': "Low",
        'Multitasking': "Medium",
        'Budget': 150000
    }

    prompt = f"""You are a python expert. You are provided an input.
    You have to check if there is a python dictionary present in the string.
    It will have the following format {user_req}.
    Your task is to just extract the relevant values from the input and return only the python dictionary in JSON format.
    The output should match the format as {user_req}.

    {delimiter}
    Make sure that the value of budget is also present in the user input. ###
    The output should contain the exact keys and values as present in the input.
    Ensure the keys and values are in the given format:
    {{
    'CPU': 'Low/Medium/High'
    'GPU': 'Low/Medium/High',
    'Display Quality':'Low/Medium/High',
    'Portability':'Low/Medium/High',
    'Multitasking':'Low/Medium/High',
    'Processing speed':'Low/Medium/High',
    'Budget':'Numerical Value without any comma'
    }}
    Here are some sample input output pairs for better understanding:
    {delimiter}
    input 1: CPU: Low - GPU: Low - Display Quality: High - Portability: Low - Multitasking: High - Budget: 50000
    output 1: {{'CPU':'Low', GPU': 'Low', 'Display Quality': 'High', 'Portability': 'Low', 'Multitasking': 'High', 'Budget': 50000}}

    input 2: CPU: Low - GPU: High - Display Quality: High - Portability: Medium - Multitasking: High - Budget: 90000
    output 2: {{'CPU':'Low', GPU': 'High', 'Display Quality': 'High', 'Portability': 'Medium', 'Multitasking': 'High', 'Budget': 90000}}

    input 2: CPU: High - GPU: High - Display Quality: High - Portability: Low - Multitasking: High - Budget: 125000
    output 2: {{'CPU':'High', GPU': 'High', 'Display Quality': 'High', 'Portability': 'Low', 'Multitasking': 'High', 'Budget': 125000}}
    {delimiter}
    """
    messages = [
        {"role": "system", "content":prompt },
        {"role": "user", "content":f"""Here is the user input: {response}""" }
    ]
    confirmation = get_chat_completions(messages, json_format = True)
    return confirmation


def product_map_layer(laptop_description):
    """
    This function will analyze the `laptop_description` and create a dictionary

    Parameters
    -------------
        laptop_description : str
        Brief description about the laptop

    Returns
    -------------
        response : dict
        Return a dictioanry consisting of all key value pairs
    
    """
    delimiter = "#####"

    lap_spec = {
        "CPU": "(CPU Performance)",
        "GPU": "(GPU Performance)",
        "Display Quality": "(Resolution of Display)",
        "Portability": "(Laptop Weight)",
        "Multitasking": "(RAM Size)",
        "Budget": "(Price of laptop)"
    }

    values = {'Low', 'Medium', 'High'}

    prompt=f"""
    You are a intelligent laptop buying assitant who is an expert in extracting features of a laptop from the provided laptop description data
    To extract features from laptop description, it is must to adhere to following steps:

    Step 1: Extract the laptop's primary features from the description as follows: {laptop_description}
    Step 2: Store the extracted features in {lap_spec} \
    Step 3: Classify each of the items in {lap_spec} into {values} based on the following rules: \

    {delimiter}
    GPU:
    - Low: <<< if GPU is an entry-level such as an integrated graphics processor or entry-level dedicated graphics like Intel UHD >>> , \n
    - Medium: <<< if mid-range dedicated graphics like M1, AMD Radeon, Intel Iris >>> , \n
    - High: <<< high-end dedicated graphics like Nvidia RTX or NVIDIA GTX series >>> , \n

    CPU:
    - Low: <<< if CPU is an entry-level processor like AMD Athlon, Intel Celeron, Intel Core 2 Duo >>> , \n
    - Medium: <<< if mid-range processor like Intel Core i3, Intel Core i5, AMD Ryzen 3 or AMD Ryzen 5 >>> , \n
    - High: <<< if a high-end processor like Intel Core i7, Intel Core i9, AMD Ryzen 7, AMD Ryzen 9 or Apple Series of Processors>>> , \n

    Display Quality:
    - Low: <<< if resolution is below Full HD (e.g., 1366x768). >>> , \n
    - Medium: <<< if Full HD resolution (1920x1080) or higher. >>> , \n
    - High: <<< if High-resolution display (e.g., 4K, Retina) with excellent color accuracy and features like HDR support. >>> \n

    Portability:
    - High: <<< if laptop weight is less than or equal to 1.0 kg >>> , \n
    - Medium: <<< if laptop weight is between 1.0 kg and 2.0 kg >>> , \n
    - Low: <<< if laptop weight is greater than or equal to 2.0 kg >>> \n
    Classifying laptops on the basis of portability has to adhere with mentioned rules and are to be be done followed very strictly

    Multitasking:
    - Low: <<< If RAM size is less than or equal to 8 GB >>> , \n
    - Medium: <<< if RAM size is between 9 GB & 16 GB >>> , \n
    - High: <<< if RAM size is greater than 20 GB >>> \n

    Budget:
    Extract the price of the laptop which is mentioned in Indian Rupee and convert the same into proper integer format so that it can be analyzed further.
    Be very careful while extracting price of the product. In case, if you are unable to extract the price mention the price as 0
    

    {delimiter}

    {delimiter}
    Here is input output pair for few-shot learning:
    input 1: "The Dell Inspiron is a versatile laptop that combines powerful performance and affordability. It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing. With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity. The laptop sports a vibrant 15.6" LCD display with a resolution of 1920x1080, delivering crisp visuals and immersive viewing experience. Weighing just 2.5 kg, it is highly portable, making it ideal for on-the-go usage. Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard for enhanced typing convenience. With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion for work or entertainment. All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users."
    output 1: {{'CPU': 'Medium', 'GPU':'Medium', 'Display Quality': 'Medium', 'Portability':'Low', 'Multitasking':'Low', 'Budget': 35000}}

    {delimiter}
    ### Strictly don't keep any other text in the values of the JSON dictionary other than Low, Medium or High else will be heavily penalized###
    """
    input = f"""Follow the above instructions step-by-step and output the dictionary in JSON format {lap_spec} for the following laptop {laptop_description}."""
    #see that we are using the Completion endpoint and not the Chatcompletion endpoint
    messages=[{"role": "system", "content":prompt },{"role": "user","content":input}]

    response = get_chat_completions(messages, json_format = True)

    return response


def filter_data_by_budget(max_budget_value=None):
    """
    This function will filter only those rows which has a budget lesser than the argument `max_budget_value` passed
    If no budget value is passed as argument, it means all rows are selected
    If the budget is selected such that after filtering no rows are present; then `False` is returned

    Parameters
    -------------
        max_budget_value : int
        This is the maximum budget selected

    Returns
    -------------
        temp_df : obj
        Returns pandas dataframe object with only those rows which has budget less than equal to `max_budget_value`
    
    """
    # pattern = r"(\d*)"
    # max_budget_value = int(re.search(pattern, max_budget_value.replace(",", "")).group())
    
    temp_df = df[df["Budget"] <= max_budget_value]
    if len(temp_df) == 0:
        return False
    else:
        return temp_df
    
    
def higher_the_better_rule(series, user_value):
    """
    This function will change specific dataframe column values as mentioned in argument `series`
    Any value above or equal to `user_value` passed as argument is marked as 1 else 0

    Parameters
    -------------
        series : obj
        Each column of pandas dataframe is passed as series

        user_value : str
        User value extracted from user for the particlular feature

    Returns
    -------------
        series : obj
        Return pandas series for that particular column
    
    """
    if user_value == "High":
        series_ = series.map({"High": 1.0, "Medium": 0.0, "Low": 0.0})
    elif user_value == "Medium":
        series_ = series.map({"High": 1.2, "Medium": 1, "Low": 0})
    elif user_value == "Low":
        series_ = series.map({"High": 1.5, "Medium": 1.2, "Low": 1})
    else:
        return False  #Inavlid user value

    return series_

def lower_the_better_rule(series, user_value):
    """
    This function will change specific dataframe column values as mentioned in argument `column_name`
    Any value below or equal to `user_value` passed as argument is marked as 1 else 0

    Parameters
    -------------
        series : obj
        Each column of pandas dataframe is passed as series

        user_value : str
        User value extracted from user for the particlular feature

    Returns
    -------------
        series : obj
        Return pandas series for that particular column
    
    """
    if user_value == "High":
        series_ = series.map({"High": 1, "Medium": 1.2, "Low": 1.5})
    elif user_value == "Medium":
        series_ = series.map({"High": 0, "Medium": 1, "Low": 1.2})
    elif user_value == "Low":
        series_ = series.map({"High": 0, "Medium": 0, "Low": 1})
    else:
        return False  #Inavlid user value

    return series_

df = pd.DataFrame(df["Description"], columns=["Description"])
df.head()

df["dict_"] = df["Description"].apply(product_map_layer)
df = pd.concat([df, df["dict_"].apply(pd.Series)], axis=1)
if "dict_" in df.columns:
    df = df.drop(columns=['dict_'])
df.head()

df_after_budget_filter = filter_data_by_budget(sample_user_query["Budget"])

df_after_budget_filter["CPU"] = higher_the_better_rule(df_after_budget_filter["CPU"], sample_user_query["CPU"])
df_after_budget_filter["GPU"] = higher_the_better_rule(df_after_budget_filter["GPU"], sample_user_query["GPU"])
df_after_budget_filter["Display Quality"] = higher_the_better_rule(df_after_budget_filter["Display Quality"], sample_user_query["Display Quality"])
df_after_budget_filter["Portability"] = lower_the_better_rule(df_after_budget_filter["Portability"], sample_user_query["Portability"])
df_after_budget_filter["Multitasking"] = higher_the_better_rule(df_after_budget_filter["Multitasking"], sample_user_query["Multitasking"])
df_after_budget_filter["Score"] = df_after_budget_filter["CPU"] + df_after_budget_filter["GPU"] + df_after_budget_filter["Display Quality"] + df_after_budget_filter["Portability"] + df_after_budget_filter["Multitasking"]

df_after_budget_filter = df_after_budget_filter.sort_values(by=["Score", "Budget"], ascending=[False, True], ignore_index=True)
df_after_budget_filter.head(3)


def initialize_conv_reco(products):
    system_message = f"""
    You are a good laptop salesman representative who is an expert in understanding user requirement in terms of buying a laptop as per user requirement
    Top recommended laptops have already been shortlisted and provided
    You need to explain specifications of those laptops with utmost accuracy and also make sure to keep user requirement in check while explaining those laptops

    Start with a brief summary of each laptop in the following format:
    1. <Laptop Name> : <Price in Indian Rupees>, <Major specifications of the laptop>
    2. <Laptop Name> : <Price in Rupees>, <Major specifications of the laptop>

    """
    user_message = f""" These are the user's products: {products}"""
    conversation = [
        {"role": "system", "content": system_message },
        {"role":"user","content":user_message}
    ]
    return conversation


def dialogue_mgmt_system():
    conversation = initialize_conversation()
    introduction = get_chat_completions(conversation)
    display(introduction + '\n')
    top_3_laptops = None
    user_input = ''
    while(user_input != "exit"):
        user_input = input("")

        moderation = moderation_check(user_input)
        if moderation == 'Flagged':
            display("Sorry, this message has been flagged. Please restart your conversation.")
            break

        if top_3_laptops is None:
            conversation.append({"role": "user", "content": user_input})
            response_assistant = get_chat_completions(conversation)
            moderation = moderation_check(response_assistant)
            if moderation == 'Flagged':
                display("Sorry, this message has been flagged. Please restart your conversation.")
                break

            confirmation = intent_confirmation_layer(response_assistant)
            if "No" in confirmation.get('result'):
                conversation.append({"role": "assistant", "content": str(response_assistant)})
                print("\n" + str(response_assistant) + "\n")

            else:
                response = dictionary_present(response_assistant)

                # print("Thank you for providing all the information. Kindly wait, while I fetch the products: \n")
                # top_3_laptops = compare_laptops_with_user(response)
                # print("Top 3 laptops are", top_3_laptops)
                # validated_reco = recommendation_validation(top_3_laptops)

                df_after_budget_filter = filter_data_by_budget(response["Budget"])
                
                df_after_budget_filter["CPU"] = higher_the_better_rule(df_after_budget_filter["CPU"], response["CPU"])
                df_after_budget_filter["GPU"] = higher_the_better_rule(df_after_budget_filter["GPU"], response["GPU"])
                df_after_budget_filter["Display Quality"] = higher_the_better_rule(df_after_budget_filter["Display Quality"], response["Display Quality"])
                df_after_budget_filter["Portability"] = lower_the_better_rule(df_after_budget_filter["Portability"], response["Portability"])
                df_after_budget_filter["Multitasking"] = higher_the_better_rule(df_after_budget_filter["Multitasking"], response["Multitasking"])
                df_after_budget_filter["Score"] = df_after_budget_filter["CPU"] + df_after_budget_filter["GPU"] + df_after_budget_filter["Display Quality"] + df_after_budget_filter["Portability"] + df_after_budget_filter["Multitasking"]
                
                df_after_budget_filter = df_after_budget_filter.sort_values(by=["Score", "Budget"], ascending=[False, True], ignore_index=True)
                
                
                conversation_reco = initialize_conv_reco(list(df_after_budget_filter.head(3)["Description"].values))
                conversation_reco.append({"role": "user", "content": "This is my user profile" + str(response)})
                recommendation = get_chat_completions(conversation_reco)
                moderation = moderation_check(recommendation)
                if moderation == 'Flagged':
                    display("Sorry, this message has been flagged. Please restart your conversation.")
                    break
                conversation_reco.append({"role": "assistant", "content": str(recommendation)})
                print(str(recommendation) + '\n')
        else:
            conversation_reco.append({"role": "user", "content": user_input})
            response_asst_reco = get_chat_completions(conversation_reco)
            moderation = moderation_check(response_asst_reco)
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break
            print('\n' + response_asst_reco + '\n')
            conversation.append({"role": "assistant", "content": response_asst_reco})


            ###########################################################################



st.title("🛍️ Shop Assist AI — Laptop Recommendation Chatbot")

# ---- Initialize session state (runs once per browser session) ----
if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_conversation()
    st.session_state.messages = []
    st.session_state.top_3_laptops = None
    st.session_state.conversation_reco = None

    intro = get_chat_completions(st.session_state.conversation)
    st.session_state.conversation.append({"role": "assistant", "content": intro})
    st.session_state.messages.append({"role": "assistant", "content": intro})

# ---- Render chat history ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Handle new user input ----
user_input = st.chat_input("Type your message here...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    moderation = moderation_check(user_input)
    if moderation == 'Flagged':
        msg = "Sorry, this message has been flagged. Please restart the conversation."
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.stop()

    # ----- Phase 1: gathering requirements -----
    if st.session_state.top_3_laptops is None:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        response_assistant = get_chat_completions(st.session_state.conversation)

        moderation = moderation_check(response_assistant)
        if moderation == 'Flagged':
            msg = "Sorry, this message has been flagged. Please restart the conversation."
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()

        confirmation = intent_confirmation_layer(response_assistant)

        if "No" in confirmation.get('result'):
            st.session_state.conversation.append({"role": "assistant", "content": response_assistant})
            with st.chat_message("assistant"):
                st.markdown(response_assistant)
            st.session_state.messages.append({"role": "assistant", "content": response_assistant})

        else:
            with st.chat_message("assistant"):
                st.markdown(response_assistant)
            st.session_state.messages.append({"role": "assistant", "content": response_assistant})

            with st.spinner("Thank you for providing all the information. Fetching the best laptops for you..."):
                response = dictionary_present(response_assistant)
                top_3_laptops = compare_laptops_with_user(response)
                validated_reco = recommendation_validation(top_3_laptops)
                conversation_reco = initialize_conv_reco(validated_reco)
                conversation_reco.append({"role": "user", "content": "This is my user profile" + str(response)})
                recommendation = get_chat_completions(conversation_reco)

            moderation = moderation_check(recommendation)
            if moderation == 'Flagged':
                msg = "Sorry, this message has been flagged. Please restart the conversation."
                with st.chat_message("assistant"):
                    st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.stop()

            conversation_reco.append({"role": "assistant", "content": recommendation})

            with st.chat_message("assistant"):
                st.markdown(recommendation)
            st.session_state.messages.append({"role": "assistant", "content": recommendation})

            st.session_state.top_3_laptops = top_3_laptops
            st.session_state.conversation_reco = conversation_reco

    # ----- Phase 2: follow-up Q&A about recommended laptops -----
    else:
        st.session_state.conversation_reco.append({"role": "user", "content": user_input})
        response_asst_reco = get_chat_completions(st.session_state.conversation_reco)

        moderation = moderation_check(response_asst_reco)
        if moderation == 'Flagged':
            msg = "Sorry, this message has been flagged. Please restart the conversation."
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()

        st.session_state.conversation_reco.append({"role": "assistant", "content": response_asst_reco})

        with st.chat_message("assistant"):
            st.markdown(response_asst_reco)
        st.session_state.messages.append({"role": "assistant", "content": response_asst_reco})