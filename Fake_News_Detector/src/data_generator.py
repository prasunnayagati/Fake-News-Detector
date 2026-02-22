import pandas as pd
import os

def create_sample_data():
    os.makedirs('data/raw', exist_ok=True)
    
    true_news = [
        {"title": "Global Climate Accord Reached", "text": "Leaders from 190 countries signed a historic agreement to reduce carbon emissions by 50% by 2030, citing Reuters reports.", "subject": "worldnews", "date": "2024-01-01"},
        {"title": "New Tech Breakthrough in Solar Energy", "text": "Researchers at MIT have developed a new solar cell that is 40% more efficient than current technologies, as reported by AP.", "subject": "tech", "date": "2024-01-02"},
        {"title": "Economic Growth Exceeds Expectations", "text": "The latest GDP figures show a 3.5% growth rate, outpacing analyst predictions of 2.8%, according to official Washington statistics.", "subject": "politics", "date": "2024-01-03"}
    ] * 50 # Duplicate to have enough for a tiny training set
    
    fake_news = [
        {"title": "Alien Invasion Imminent Says Top Secret Source", "text": "A leaked document from a secret underground base claims that aliens are planning to invade Earth next Tuesday. Share before it's deleted!", "subject": "conspiracy", "date": "2024-01-01"},
        {"title": "Magic Pill Cures All Diseases Instantly", "text": "Big Pharma is hiding the truth about a $1 pill that can cure anything from the common cold to cancer. Doctors hate this one trick!", "subject": "health", "date": "2024-01-02"},
        {"title": "Celebrity Replaced by Clone, Evidence Found", "text": "Fans noticed a change in the star's earlobe shape, proving they have been replaced by a government-funded biological clone.", "subject": "gossip", "date": "2024-01-03"}
    ] * 50
    
    pd.DataFrame(true_news).to_csv('data/raw/True.csv', index=False)
    pd.DataFrame(fake_news).to_csv('data/raw/Fake.csv', index=False)
    print("Sample data generated in data/raw/")

if __name__ == "__main__":
    create_sample_data()
