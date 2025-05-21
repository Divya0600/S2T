import sys
from api.ollama_handler import generate_response  # Import function from ollama_handler.py

# Function to generate a structured Minutes of Meeting (MOM) document
def generate_mom(transcript):
    system_prompt = (
        "Generate a structured Minutes of Meeting (MOM) document based on the following meeting transcript. The MOM should include:\n\n"
        "• An Agenda outlining discussion topics.\n"
        "• Detailed discussion points, decisions, and key takeaways.\n"
        "• Action items in a tabular format with assigned owners and deadlines.\n\n"
        "Ensure clarity, conciseness, and professional formatting.\n\n"
        "Meeting Transcript:\n"
    )

    print("\n📝 Generating MOM...\n")
    
    # Directly call Ollama API and get the response
    response_text = generate_response(prompt=transcript, system_message=system_prompt, max_tokens=1500)
    
    print("\n📌 **Minutes of Meeting (MOM):**\n", response_text, "\n")  # Print final output

# Main loop for user input
try:
    while True:
        print("\n📥 Paste your meeting transcript below (or type 'exit' to quit):")
        transcript = input("\n")
        
        if transcript.lower() in ["exit", "quit"]:
            print("👋 Exiting MOM Generator. See you next time!")
            sys.exit(0)

        generate_mom(transcript)

except KeyboardInterrupt:
    print("\n👋 Exiting MOM Generator. See you next time!")
