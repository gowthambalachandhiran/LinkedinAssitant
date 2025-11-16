from typing import List, Dict
import pandas as pd
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.agents import Tool
from langchain.tools import BaseTool
from crewai import Agent, Task, Crew,LLM
import os
from dotenv import load_dotenv

class AIContentProcessor:
    def __init__(self, llm):
        self.llm = llm
        
    def filter_ai_content(self, twitter_df: pd.DataFrame, reddit_df: pd.DataFrame) -> Dict:
        """Filter and combine AI-related content from both platforms"""
        
        # Process Twitter content
        twitter_content = []
        for _, row in twitter_df.iterrows():
            content_data = {
                'text': row['tweettext'],
                'author': row['author'],
                'date': row['date'],
                'platform': 'Twitter',
                'url': f"https://twitter.com/{row['author']}/status/{row['id']}" if 'id' in row else None
            }
            twitter_content.append(content_data)
            
        # Process Reddit content - simplified for subreddit, text, and created_at
        reddit_content = []
        for _, row in reddit_df.iterrows():
            content_data = {
                'text': row['content'],
                'subreddit': row['subreddit'],
                'date': row['created_utc'],
                'platform': 'Reddit',
            }
            reddit_content.append(content_data)
            
        return {
            'twitter': twitter_content,
            'reddit': reddit_content
        }

def create_ai_news_crew(llm, filtered_content: Dict) -> Crew:
    """Create a crew of agents for content analysis and article generation"""
    
    # Content Analyzer Agent
    analyzer = Agent(
        role='AI News Analyzer',
        goal='Identify and analyze significant AI developments and news from social media content',
        backstory="""Expert in AI technology with deep understanding of the field. 
                    Skilled at identifying genuine AI developments while filtering out 
                    speculation and controversial content.""",
        llm=llm,
        verbose=True
    )
    
    # Article Writer Agent
    writer = Agent(
        role='LinkedIn Article Writer',
        goal='Create professional, well-referenced LinkedIn articles about AI developments',
        backstory="""Professional tech writer specializing in AI content. 
                    Skilled at creating engaging, factual articles with proper attribution 
                    and clear references.""",
        llm=llm,
        verbose=True
    )
    
    # Analysis Task
    analysis_task = Task(
        description=f"""
        Analyze the provided social media content and:
        1. Identify significant AI developments and news
        2. Filter out controversial or speculative content
        3. Group related information by topics
        4. Verify information across multiple sources when available
        
        Content to analyze: {filtered_content}
        
        Focus on:
        - New AI technology releases
        - Research breakthroughs
        - Industry updates
        - Educational resources
        - Development tools and frameworks
        
        Exclude:
        - Controversial AI discussions
        - Unverified claims
        - Personal opinions without backing
        - Political content
        """,
        agent=analyzer,
        expected_output="""Structured analysis of AI developments including:
            - Key trends and announcements
            - Verified developments
            - Sources and references
            - Grouped themes and topics
            - References"""
    )
    
    # Writing Task
    writing_task = Task(
        description="""
        Create a professional LinkedIn article that:
        1. Presents key AI developments and news clearly
        2. Includes proper attribution with links to sources
        3. Maintains professional tone
        4. Organizes information logically
        5. Adds valuable context when needed
        
        Format:
        - Title: Clear and specific to the main developments
        - Introduction: Overview of key points
        - Body: Organized by topics/developments
        - Each point includes source reference
        - Conclusion: Summary of implications
        
        
        Length: 400-600 words
        """,
        agent=writer,
        expected_output="""A professional LinkedIn article with:
            - Clear title
            - Structured sections
            - Source attributions
            - Key AI developments
            - Professional tone""",
        context=[analysis_task]
    )
    
    return Crew(
        agents=[analyzer, writer],
        tasks=[analysis_task, writing_task],
        process="sequential",
        verbose=True
    )

def generate_article(twitter_df: pd.DataFrame, reddit_df: pd.DataFrame, llm) -> str:
    """Generate LinkedIn article from social media content"""
    
    # Initialize content processor
    processor = AIContentProcessor(llm)
    
    # Filter and combine content
    filtered_content = processor.filter_ai_content(twitter_df, reddit_df)
    
    # Create and run the crew
    crew = create_ai_news_crew(llm, filtered_content)
    result = crew.kickoff()
    
    # Save the article
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"ai_linkedin_article_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(result))
    
    return str(result), filename

def setup_llm():
    """Initialize LLM using CrewAI's LLM wrapper"""
    try:
        return LLM(
            model="gemini/gemini-pro",  # Changed to match the template format
            temperature=0.7
        )
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        raise

def load_data(twitter_file: str, reddit_file: str) -> tuple:
    """Load and validate data from CSV files"""
    try:
        twitter_df = pd.read_csv(twitter_file,encoding='ISO-8859-1')
        reddit_df = pd.read_csv(reddit_file,encoding='ISO-8859-1')
        
        # Validate Twitter DataFrame columns
        required_twitter_cols = ['tweettext', 'author', 'date']
        if not all(col in twitter_df.columns for col in required_twitter_cols):
            raise ValueError(f"Twitter CSV must contain columns: {required_twitter_cols}")
            
        # Validate Reddit DataFrame columns - simplified
        required_reddit_cols = ['subreddit', 'content', 'created_utc']
        if not all(col in reddit_df.columns for col in required_reddit_cols):
            raise ValueError(f"Reddit CSV must contain columns: {required_reddit_cols}")
        
        return twitter_df, reddit_df
        
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {str(e)}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: One or both input files are empty")
        raise
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def main():
    """Main execution function"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for required environment variables
        if not os.getenv("GEMINI_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY not found in environment variables")
        
        # Initialize LLM
        print("Initializing LLM...")
        llm = setup_llm()
        
        # Define input files
        twitter_file = "twitter_data.csv"
        reddit_file = "reddit_data.csv"
        
        # Load and validate data
        print("Loading data from CSV files...")
        twitter_df, reddit_df = load_data(twitter_file, reddit_file)
        
        print(f"Loaded {len(twitter_df)} tweets and {len(reddit_df)} Reddit posts")
        
        # Generate article
        print("Generating LinkedIn article...")
        article_text, filename = generate_article(twitter_df, reddit_df, llm)
        
        # Print results
        print(f"\nArticle successfully generated and saved to: {filename}")
        print("\nArticle Preview:")
        print("="*50)
        preview_length = 500
        print(article_text[:preview_length] + "..." if len(article_text) > preview_length else article_text)
        print("="*50)
        
        return filename
        
    except Exception as e:
        print(f"Fatal error in main execution: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()