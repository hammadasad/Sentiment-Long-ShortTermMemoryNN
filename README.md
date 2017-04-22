# Sentiment-Long-ShortTermMemoryNN

Architecture:

Input -> Embeddings -> LSTM -> Sigmoid
   
   - Used embeddings to prevent one-hot encoding of 1000s of classes
   - Used LSTM Cells to prevent vanishing gradients
   - Validation Accuracy ~85% at 10 epochs
   - Test Accuracy ~83%
   
   
   
