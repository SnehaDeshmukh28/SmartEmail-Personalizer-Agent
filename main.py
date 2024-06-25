import os
import time
import csv
from crewai import Crew
from langchain_groq import ChatGroq
from agents import EmailPersonalizationAgents
from tasks import PersonalizeEmailTask

# 0. Setup environment
from dotenv import load_dotenv
load_dotenv()

email_template = """
Hey [Name]!

I regularly post updates on my projects on LinkedIn and GitHub, so be sure to follow me there for the 
latest news and insights into my work.

If you have any questions or need assistance with your projects, feel free to pin me on LinkedIn to 
schedule a quick call. I am always happy to connect, collaborate, and provide support.

I work on various AI projects, so stay tuned for updates and new developments.

Looking forward to connecting with you!

Best regards,
Sneha Deshmukh
"""

# 1. Create agents
agents = EmailPersonalizationAgents()

email_personalizer = agents.personalize_email_agent()
ghostwriter = agents.ghostwriter_agent()

# 2. Create tasks
tasks = PersonalizeEmailTask()

personalize_email_tasks = []
ghostwrite_email_tasks = []

# Path to the CSV file containing client information
csv_file_path = 'data/clients_small.csv'

# Open the CSV file
with open(csv_file_path, mode='r', newline='') as file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(file)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Access each field in the row
        recipient = {
            'first_name': row['first_name'],
            'last_name': row['last_name'],
            'email': row['email'],
            'bio': row['bio'],
            'last_conversation': row['last_conversation']
        }

        # Create a personalize_email task for each recipient
        personalize_email_task = tasks.personalize_email(
            agent=email_personalizer,
            recipient=recipient,
            email_template=email_template
        )

        # Create a ghostwrite_email task for each recipient
        ghostwrite_email_task = tasks.ghostwrite_email(
            agent=ghostwriter,
            draft_email=personalize_email_task,
            recipient=recipient
        )

        # Add the task to the crew
        personalize_email_tasks.append(personalize_email_task)
        ghostwrite_email_tasks.append(ghostwrite_email_task)


# Setup Crew
crew = Crew(
    agents=[
        email_personalizer,
        ghostwriter
    ],
    tasks=[
        *personalize_email_tasks,
        *ghostwrite_email_tasks
    ],
    max_rpm=29
)

# Kick off the crew
start_time = time.time()

results = crew.kickoff()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Crew kickoff took {elapsed_time} seconds.")
print("Crew usage", crew.usage_metrics)
