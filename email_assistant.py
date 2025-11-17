from openai import OpenAI
from pydantic import BaseModel

class EmailRequest(BaseModel):
    email: str
    tone: str

def rewrite_email(request: EmailRequest, openai_client: OpenAI):
    """Receives an email text and rewrites it."""
    try:
        email = request.email
        email_tone = request.tone

        system_prompt = f"""
        I want you to act as an email assistant. I want you to rewrite the email text provided in a chosen tone.
        
        Email Text:
        {email}

        The tone of the email is: {email_tone}.

        Rules:
        - I want you to write an email in the tone that is given.
        - Use a basic email template. Make the email structure fit the tone.
        """

        try:
            openai_response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.5,
            )
            rewritten_email = openai_response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            rewritten_email = "I encountered an error while processing your question."
        return {"rewritten_email": rewritten_email}
    except Exception as e:
        print(f"Error in email rewrite: {str(e)}")
        raise