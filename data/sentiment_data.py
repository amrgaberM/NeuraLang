# Sentiment analysis dataset with varied examples

sentences = [
    # Positive sentiment
    "I love this product, it is amazing!",
    "Absolutely fantastic experience!",
    "I feel happy and excited today.",
    "I enjoy spending time with my friends.",
    "Thrilled to be part of this event.",
    "Best purchase I've made this year!",
    "The customer service was excellent and helpful.",
    "What a wonderful day at the beach!",
    "I'm so grateful for all the support.",
    "This restaurant serves delicious food.",
    "Great quality and fast delivery.",
    "I highly recommend this to everyone.",
    "The movie was entertaining and fun.",
    "I'm really impressed with the results.",
    "Such a beautiful and peaceful place.",
    
    # Negative sentiment
    "This is the worst movie I have ever seen.",
    "I am very disappointed with the service.",
    "Not what I expected, really bad.",
    "This food tastes awful.",
    "I will never buy this again.",
    "Terrible experience, complete waste of money.",
    "The product broke after one day.",
    "Very poor customer support, no help at all.",
    "I'm frustrated with how long this took.",
    "Absolutely horrible quality for the price.",
    "This app crashes constantly, very annoying.",
    "I regret making this purchase.",
    "The hotel room was dirty and uncomfortable.",
    "Worst decision I've made in a while.",
    "I'm extremely unhappy with the outcome.",
    
    # Neutral (optional for more advanced models)
    "The package arrived on Tuesday.",
    "It is what it is, nothing special.",
    "The item matches the description.",
    "I received the product as expected.",
    "Standard quality, meets basic requirements."
]

labels = [
    # Positive labels
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    
    # Negative labels
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    
    # Neutral labels
    "neutral", "neutral", "neutral", "neutral", "neutral"
]