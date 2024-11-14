import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import nltk
from nltk.stem import WordNetLemmatizer
import re

# --- 📔 Conversation State Class ---
class ConversationState:
    def __init__(self):
        # 🧠 How Conversation State Works:
        #   - Tracks the entire state of a conversation session
        #   - Each attribute serves a specific purpose:
        
        # Current intent being processed (e.g., "transfer_money", "balance_enquiry")
        self.current_intent: Optional[str] = None
        
        # Dictionary storing extracted information (e.g., amount, recipient)
        self.entities: Dict = {}
        
        # Timestamp of last user interaction (for timeout handling)
        self.last_interaction: datetime = datetime.now()
        
        # List storing all previous interactions and their details
        self.context_history: List[Dict] = []
        
        # Information that still needs to be collected
        self.pending_info: Dict = {}
        
        # Tracks which step we're at in the conversation
        self.conversation_step: int = 0
        
        # What specific information we're waiting for from user
        self.awaiting_entity: Optional[str] = None
        
        # Whether we're waiting for user to confirm an action
        self.awaiting_confirmation: bool = False
        
        # Whether a transfer has been completed (for transfer intent)
        self.transfer_completed: bool = False

# --- 📔 Main Conversation Manager Class ---
class ConversationManager:
    def __init__(self, model, intents, words, classes):
        # 🧠 How Initialisation Works:
        #   - Takes trained model from first file
        #   - Stores reference to intents, vocabulary, and classes
        #   - Sets up conversation management tools
        
        # Store model and training data
        self.model = model
        self.intents = intents
        self.words = words
        self.classes = classes
        
        # Dictionary to store active conversations
        self.sessions: Dict[str, ConversationState] = {}
        
        # Session timeout (30 minutes of inactivity)
        self.session_timeout = timedelta(minutes=30)
        
        # Initialise lemmatizer for word processing
        self.lemmatizer = WordNetLemmatizer()
        
        # --- 📔 Conversation Flow Definitions ---
        # 🧠 How Follow-up Mapping Works:
        #   - Defines the required information for each intent
        #   - Specifies follow-up questions for missing information
        #   - Contains confirmation messages and success/failure responses
        self.follow_up_mapping = {
            # Balance Enquiry Flow
            'balance_enquiry': {
                'required_entities': ['account_number'],
                'follow_up_questions': {
                    'missing_account': "What's your account number?",
                }
            },
            # Money Transfer Flow
            'transfer_money': {
                'required_entities': ['recipient', 'amount'],
                'follow_up_questions': {
                    'missing_recipient': "Who would you like to send money to?",
                    'missing_amount': "How much would you like to transfer?",
                    'confirm_transfer': "Would you like to transfer {amount} to {recipient}? (Reply 'yes' to confirm or 'no' to cancel)",
                    'transfer_confirmed': "Transfer of {amount} to {recipient} has been completed successfully. Is there anything else I can help you with?",
                    'transfer_cancelled': "Transfer has been cancelled. Is there anything else I can help you with?"
                }
            },
            # Loan Enquiry Flow
            'loan_enquiry': {
                'required_entities': ['loan_type', 'amount'],
                'follow_up_questions': {
                    'missing_loan_type': "What type of loan are you interested in? (personal, home, or business)",
                    'missing_amount': "How much would you like to borrow?"
                }
            }
        }

    # --- 📔 Session Management Methods ---
    def create_session(self, session_id: str) -> None:
        # 🧠 How Session Creation Works:
        #   - Creates new conversation state for given session ID
        #   - Used for:
        #     - New conversations
        #     - Resetting timed-out sessions
        #   - Initialises all tracking variables to default state
        self.sessions[session_id] = ConversationState()

    def get_session(self, session_id: str) -> Optional[ConversationState]:
        # 🧠 How Session Retrieval Works:
        #   - Gets existing session or creates new one
        #   - Ensures every conversation has an active session
        #   - Returns ConversationState object for tracking interaction
        if session_id not in self.sessions:
            self.create_session(session_id)
        return self.sessions[session_id]

    def reset_session_state(self, session: ConversationState) -> None:
        # 🧠 How Session Reset Works:
        #   - Clears all tracking variables after completing transaction
        #   - Like wiping a slate clean for new conversation
        #   Process:
        #     1. Clear current intent
        #     2. Clear collected entities
        #     3. Reset all flags and counters
        session.current_intent = None
        session.entities = {}
        session.awaiting_entity = None
        session.awaiting_confirmation = False
        session.transfer_completed = False
        session.conversation_step = 0

    def is_confirmation(self, message: str) -> bool:
        # 🧠 How Confirmation Detection Works:
        #   - Checks if user message is a positive confirmation
        #   - Recognises various ways of saying "yes"
        #   - Case insensitive and handles extra whitespace
        confirmations = {'yes', 'confirm', 'correct', 'right', 'sure', 'ok', 'okay', 'yep', 'yeah', 'y'}
        return message.lower().strip() in confirmations

    def is_denial(self, message: str) -> bool:
        # 🧠 How Denial Detection Works:
        #   - Checks if user message is a negative response
        #   - Recognises various ways of saying "no"
        #   - Case insensitive and handles extra whitespace
        denials = {'no', 'cancel', 'wrong', 'incorrect', 'nope', 'nah', 'n'}
        return message.lower().strip() in denials

    def handle_confirmation(self, session: ConversationState, message: str) -> str:
        # 🧠 How Confirmation Handling Works:
        #   - Processes user's response to confirmation request
        #   - Three possible paths:
        #     1. User confirms -> Complete transaction
        #     2. User denies -> Cancel transaction
        #     3. Unclear response -> Ask again
        
        if self.is_confirmation(message):
            # Handle positive confirmation
            session.transfer_completed = True
            session.awaiting_confirmation = False
            response = self.follow_up_mapping['transfer_money']['follow_up_questions']['transfer_confirmed'].format(
                amount=session.entities.get('amount', ''),
                recipient=session.entities.get('recipient', '')
            )
            self.reset_session_state(session)
            return response
            
        elif self.is_denial(message):
            # Handle denial/cancellation
            session.awaiting_confirmation = False
            response = self.follow_up_mapping['transfer_money']['follow_up_questions']['transfer_cancelled']
            self.reset_session_state(session)
            return response
            
        else:
            # Handle unclear response by asking again
            return self.follow_up_mapping['transfer_money']['follow_up_questions']['confirm_transfer'].format(
                amount=session.entities.get('amount', ''),
                recipient=session.entities.get('recipient', '')
            )

    def extract_entities(self, message: str, intent: str) -> Dict:
        # 🧠 How Entity Extraction Works:
        #   - Pulls out important information based on intent type
        #   - Uses regex patterns to find specific data
        #   - Different extraction rules for different intents
        entities = {}
        
        if intent == 'transfer_money':
            # 🧠 Money Transfer Entity Extraction:
            #   - Looks for two key pieces:
            #     1. Amount of money
            #     2. Recipient name
            
            # Extract amount using regex pattern
            # Matches formats: $100, 100.00, 100
            amount_pattern = r'\$?\d+(?:\.\d{2})?|\d+'
            amounts = re.findall(amount_pattern, message)
            if amounts:
                entities['amount'] = amounts[0]
            
            # Extract recipient name
            # Looks for text after "to" or "for"
            recipient_indicators = ['to', 'for']
            words = message.lower().split()
            for i, word in enumerate(words):
                if word in recipient_indicators and i + 1 < len(words):
                    entities['recipient'] = ' '.join(words[i+1:])
                    break
                    
        elif intent == 'balance_enquiry':
            # 🧠 Balance Enquiry Entity Extraction:
            #   - Looks for account numbers
            #   - Expects 10-12 digit number
            account_pattern = r'\b\d{10,12}\b'
            account_numbers = re.findall(account_pattern, message)
            if account_numbers:
                entities['account_number'] = account_numbers[0]
                
        return entities

    def get_next_required_entity(self, session_id: str) -> Optional[str]:
        # 🧠 How Required Entity Check Works:
        #   - Checks what information is still needed
        #   - Based on intent requirements
        #   Process:
        #     1. Get current session
        #     2. Check intent requirements
        #     3. Return first missing requirement
        session = self.get_session(session_id)
        if not session.current_intent:
            return None
            
        required_entities = self.follow_up_mapping.get(
            session.current_intent, {}
        ).get('required_entities', [])
        
        for entity in required_entities:
            if entity not in session.entities:
                return entity
        return None

    def clean_up_sentence(self, sentence: str) -> List[str]:
        # 🧠 How Sentence Cleanup Works:
        #   - Prepares text for processing
        #   - Same process used in training:
        #     1. Split into words (tokenize)
        #     2. Convert to lowercase
        #     3. Reduce to base form (lemmatize)
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) 
                        for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence: str) -> np.ndarray:
        # 🧠 How Bag of Words Conversion Works:
        #   - Converts text to numerical format
        #   - Creates binary vector:
        #     - 1 = word is present
        #     - 0 = word is absent
        #   Process:
        #     1. Clean up sentence
        #     2. Initialise zero vector
        #     3. Mark present words with 1
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)
        
    def predict_class(self, sentence: str) -> List[Dict]:
        # 🧠 How Intent Prediction Works:
        #   - Uses trained model to identify intent
        #   Process:
        #     1. Convert sentence to bag of words
        #     2. Get model predictions
        #     3. Filter predictions above threshold
        #     4. Sort by confidence
        
        # Convert to bag of words
        bow = self.bag_of_words(sentence)
        
        # Get model predictions
        res = self.model.predict(np.array([bow]))[0]
        
        # Filter and sort predictions
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Format results
        return_list = []
        for r in results:
            return_list.append({
                'intent': self.classes[r[0]],
                'probability': str(r[1])
            })
        return return_list

    def generate_response(self, session_id: str, user_message: str) -> str:
        # 🧠 How Response Generation Works:
        #   - Heart of the chatbot's conversation logic
        #   - Manages entire conversation flow
        #   Process:
        #     1. Get/create session
        #     2. Handle timeouts
        #     3. Process message
        #     4. Generate appropriate response
        
        # Get or create session
        session = self.get_session(session_id)
        
        # --- 📔 Timeout Handling ---
        # Check if session has timed out (30 minutes)
        if datetime.now() - session.last_interaction > self.session_timeout:
            self.create_session(session_id)
            return "Welcome back! How can I help you today?"

        # Update last interaction time
        session.last_interaction = datetime.now()

        # --- 📔 Confirmation Handling ---
        # If waiting for user to confirm something
        if session.awaiting_confirmation:
            return self.handle_confirmation(session, user_message)

        # --- 📔 Intent Processing ---
        # Get intent predictions for user message
        ints = self.predict_class(user_message)
        current_intent = ints[0]['intent'] if ints else None

        # --- 📔 Intent and Entity Management ---
        # Handle new intent or continue existing conversation
        if current_intent and not session.awaiting_entity:
            # New intent detected
            session.current_intent = current_intent
            # Extract any entities from message
            new_entities = self.extract_entities(user_message, current_intent)
            session.entities.update(new_entities)
            
        # Handle awaited entity responses
        elif session.awaiting_entity:
            if session.awaiting_entity == 'amount':
                # Extract amount from response
                try:
                    amount = ''.join(filter(str.isdigit, user_message))
                    if amount:
                        session.entities['amount'] = amount
                        session.awaiting_entity = None
                except ValueError:
                    pass
            elif session.awaiting_entity == 'recipient':
                # Store recipient name
                session.entities['recipient'] = user_message.strip()
                session.awaiting_entity = None

        # --- 📔 Context History Tracking ---
        # Update conversation history
        session.context_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'intent': session.current_intent,
            'entities': session.entities.copy()
        })

        # --- 📔 Transfer Money Flow Handling ---
        # Special handling for money transfers
        if session.current_intent == 'transfer_money':
            if 'amount' in session.entities and 'recipient' in session.entities \
                and not session.awaiting_confirmation:
                # All information collected, ask for confirmation
                session.awaiting_confirmation = True
                return self.follow_up_mapping['transfer_money']['follow_up_questions']['confirm_transfer'].format(
                    amount=session.entities['amount'],
                    recipient=session.entities['recipient']
                )

        # --- 📔 Missing Information Handling ---
        # Check if we need more information
        missing_entity = self.get_next_required_entity(session_id)
        if missing_entity:
            # Set what we're waiting for
            session.awaiting_entity = missing_entity
            # Get appropriate follow-up question
            follow_up_key = f'missing_{missing_entity}'
            follow_up_questions = self.follow_up_mapping.get(
                session.current_intent, {}
            ).get('follow_up_questions', {})
            if follow_up_key in follow_up_questions:
                return follow_up_questions[follow_up_key].format(**session.entities)

        # --- 📔 Default Response Handling ---
        # If no special handling needed, get standard response
        if not session.current_intent:
            return self.get_standard_response(ints)
        return self.get_standard_response([{'intent': session.current_intent}])

    def get_standard_response(self, ints: List[Dict]) -> str:
        # 🧠 How Standard Response Works:
        #   - Fallback for simple responses
        #   - Uses predefined responses from intents file
        #   Process:
        #     1. Check if we have valid intents
        #     2. Find matching intent in intents file
        #     3. Return random response for that intent
        
        # Handle no intents case
        if not ints:
            return "I'm not sure I understand. Could you please rephrase that?"
        
        # Get intent tag
        tag = ints[0]['intent']
        
        # Find matching intent and return random response
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])
        
        # Fallback response
        return "I'm not sure how to respond to that."