# llama_chat.py
import ollama
import json
from datetime import datetime

class LlamaNewsBot:
    def __init__(self, vector_db, model_name="llama3.1:8b"):
        self.vector_db = vector_db
        self.model_name = model_name
        self.conversation_history = {}
        
        # Test if Ollama and model are available
        try:
            # First, try to directly test the model instead of relying on list()
            try:
                test_response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": "test"}],
                    options={"max_tokens": 1}
                )
                print("✅ Ollama connection successful")
                print(f"✅ Model {model_name} ready")
            except Exception as model_error:
                # If direct test fails, try to get the model list for debugging
                print("✅ Ollama connection successful")
                try:
                    response = ollama.list()

                    # Check if model is available - improved logic
                    if 'models' in response:
                        models = response['models']
                        model_names = []
                        for model in models:
                            try:
                                if isinstance(model, dict) and 'name' in model and model['name']:
                                    model_names.append(model['name'])
                                elif isinstance(model, str) and model.strip():
                                    model_names.append(model)
                            except Exception:
                                continue

                        # Filter out empty strings and None values
                        model_names = [name for name in model_names if name and name.strip()]

                        if model_name in model_names:
                            print(f"✅ Model {model_name} ready")
                        else:
                            print(f"⚠️  Model {model_name} not found in list.")
                            print(f"Available models: {model_names}")
                            print(f"📥 Run: ollama pull {model_name}")
                    else:
                        print("⚠️  Could not retrieve model list")
                        print(f"📥 Run: ollama pull {model_name}")
                except:
                    print(f"⚠️  Model {model_name} not available")
                    print(f"📥 Run: ollama pull {model_name}")

        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("Please install Ollama and run: ollama pull llama3.1:8b")

    
    def create_context_from_search(self, search_results, max_context_length=2000):
        """Create context from search results for Llama"""
        if not search_results['documents'][0]:
            return "No relevant articles found in the database."
        
        context_parts = []
        total_length = 0
        
        for i, (doc, meta) in enumerate(zip(search_results['documents'][0], 
                                           search_results['metadatas'][0])):
            if total_length >= max_context_length:
                break
                
            article_info = f"""
Article {i+1}:
Title: {meta['title']}
Source: {meta['source']} ({meta['sector']})
Content: {doc[:400]}...
"""
            if total_length + len(article_info) <= max_context_length:
                context_parts.append(article_info)
                total_length += len(article_info)
        
        return "\n".join(context_parts)
    
    def generate_response(self, user_query, user_id="default"):
        """Generate response using Llama 3.1 8B with retrieved context"""
        
        # Search for relevant articles
        search_results = self.vector_db.search_relevant_articles(user_query, n_results=3)
        context = self.create_context_from_search(search_results)
        
        # Create system prompt
        system_prompt = f"""You are a knowledgeable news assistant. Use the following recent news articles to answer the user's question. Be conversational, informative, and cite the sources when relevant.

Recent News Context:
{context}

Guidelines:
- Provide accurate information based on the articles above
- Be conversational and engaging
- If the articles don't contain relevant information, say so politely
- Mention sources when discussing specific information
- Keep responses focused and not too lengthy"""
        
        # Create conversation prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        try:
            # Generate response with Llama 3.1 8B
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            
            bot_response = response['message']['content']
            
            # Store conversation history
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            
            self.conversation_history[user_id].append({
                'user': user_query,
                'bot': bot_response,
                'timestamp': datetime.now().isoformat(),
                'sources_used': len(search_results['documents'][0])
            })
            
            return bot_response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I'm having trouble generating a response right now. Please make sure Ollama is running and llama3.1:8b is installed."
    
    def get_conversation_stats(self):
        """Get conversation statistics"""
        total_conversations = sum(len(conv) for conv in self.conversation_history.values())
        users = len(self.conversation_history)
        
        return {
            'total_conversations': total_conversations,
            'active_users': users,
            'model': self.model_name
        }
