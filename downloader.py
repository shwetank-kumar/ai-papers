import csv
import arxiv
import os
import time
import sqlite3
from typing import List, Dict
from datetime import datetime
import argparse
import re
from fuzzywuzzy import fuzz

class Downloader:
    def __init__(self, db_path: str = 'paper_summaries.db'):
        self.db_path = db_path
        self.create_database()

    def create_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS hf_email
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      email_date TEXT,
                      title TEXT,
                      upvotes INTEGER,
                      arxiv_id TEXT,
                      downloaded INTEGER,
                      summarized INTEGER,
                      post_generated INTEGER)''')
        conn.commit()
        conn.close()

    def parse_csv(self, file_path: str) -> List[Dict[str, str]]:
        all_papers = []
        current_date = None
        
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                if row[0].startswith('Date:'):
                    # current_date = row[0].split(':', 1)[1].strip()
                    # Convert date from dd-mm-yyyy to yyyy-mm-dd
                    date_str = row[0].split(':', 1)[1].strip()
                    try:
                        date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                        current_date = date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        print(f"Warning: Invalid date format: {date_str}")
                        current_date = None
                elif row[0] == 'Paper Title' and len(row) > 1 and row[1] == 'Upvotes':
                    continue  # Skip header row
                elif len(row) == 2:
                    title, upvotes = row
                    try:
                        upvotes = int(upvotes)
                        if current_date:
                            all_papers.append({
                                "email_date": current_date,
                                "title": title.strip(),
                                "upvotes": upvotes
                            })
                        else:
                            print(f"Warning: Found paper without a date: {title}")
                    except ValueError:
                        print(f"Warning: Invalid upvote value for paper: {title}")
        
        if not all_papers:
            raise ValueError("No valid data found in the CSV file.")
        
        return all_papers
    
    def store_csv_data(self, papers: List[Dict[str, str]]):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for paper in papers:
            # First, try to update an existing row
            c.execute('''UPDATE hf_email 
                        SET email_date = ?, upvotes = ?
                        WHERE title = ?''', 
                    (paper['email_date'], paper['upvotes'], paper['title']))
            
            # If no row was updated (i.e., the paper doesn't exist), insert a new row
            if c.rowcount == 0:
                c.execute('''INSERT INTO hf_email 
                            (email_date, title, upvotes, downloaded, summarized, post_generated)
                            VALUES (?, ?, ?, 0, 0, 0)''', 
                        (paper['email_date'], paper['title'], paper['upvotes']))
        conn.commit()
        conn.close()

    def prepare_title_for_search(self, title: str) -> str:
        # Remove special characters and common words
        words = re.findall(r'\b\w+\b', title.lower())
        stop_words = {'the', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 'of', 'a', 'an'}
        important_words = [word for word in words if len(word) > 2 and word not in stop_words]
        return ' '.join(important_words[:7])  # Use up to 7 important words for the search

    def search_arxiv(self, title: str) -> arxiv.Result:
        client = arxiv.Client()
        
        # Prepare the title for search
        search_title = self.prepare_title_for_search(title)
        
        # Use a more flexible search
        search = arxiv.Search(
            query=search_title,
            max_results=10,  # Increase max results to find potential matches
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(client.results(search))
        
        # Find the best match among the results
        best_match = self.find_best_match(title, results)
        return best_match

    def find_best_match(self, original_title: str, results: List[arxiv.Result]) -> arxiv.Result:
        if not results:
            return None

        best_match = None
        highest_similarity = 0

        for result in results:
            # Use fuzzy string matching to compare titles
            similarity = fuzz.token_sort_ratio(original_title.lower(), result.title.lower())

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = result

        # Only return a match if it's reasonably similar (you can adjust this threshold)
        return best_match if highest_similarity > 60 else None
    
    def download_paper(self, paper: arxiv.Result, output_dir: str):
        arxiv_id = paper.get_short_id()
        filename = f"{arxiv_id}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        paper.download_pdf(filename=filepath)
        print(f"Downloaded: {arxiv_id} - {paper.title}")

        # Update the database to mark the paper as downloaded and not summarized
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''UPDATE hf_email SET arxiv_id = ?, downloaded = 1, summarized = 0, post_generated = 0
             WHERE title = ?''', (arxiv_id, paper.title))
        conn.commit()
        conn.close()

    def is_paper_downloaded(self, title: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT downloaded FROM hf_email WHERE title = ?", (title,))
        result = c.fetchone()
        conn.close()
        return result is not None and result[0] == 1

    def process_papers(self, file_path: str, output_dir: str):
        try:
            all_papers = self.parse_csv(file_path)
            self.store_csv_data(all_papers)
        except Exception as e:
            print(f"Error parsing CSV file: {e}")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Sort all papers by upvotes (descending)
        sorted_papers = sorted(all_papers, key=lambda x: x['upvotes'], reverse=True)

        # Get the upvotes of the 10th paper (or last paper if less than 10)
        threshold_upvotes = sorted_papers[min(9, len(sorted_papers) - 1)]['upvotes']

        # Select all papers with upvotes greater than or equal to the threshold
        top_papers = [paper for paper in sorted_papers if paper['upvotes'] >= threshold_upvotes]

        print(f"\nProcessing top papers (including ties) across all dates:")
        print(f"Number of papers to process: {len(top_papers)}")
        for paper in top_papers:
            try:
                if self.is_paper_downloaded(paper['title']):
                    print(f"Paper already downloaded: {paper['title']}")
                    continue

                print(f"Searching for: {paper['title']} (Upvotes: {paper['upvotes']}, Date: {paper['email_date']})")
                arxiv_paper = self.search_arxiv(paper['title'])
                if arxiv_paper:
                    print(f"Match found: {arxiv_paper.title}")
                    self.download_paper(arxiv_paper, output_dir)
                else:
                    print(f"No close match found (skipping): {paper['title']}")
                time.sleep(3)  # Add a delay to avoid hitting rate limits
            except Exception as e:
                print(f"Error processing paper '{paper['title']}': {e}")          
                
def main():
    parser = argparse.ArgumentParser(description="Download top papers from a CSV file and store data in SQLite.")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("--output_dir", default="downloaded_papers", help="Directory to store downloaded papers")
    parser.add_argument("--db_path", default="paper_summaries.db", help="Path to the SQLite database file")
    args = parser.parse_args()

    downloader = Downloader(db_path=args.db_path)
    downloader.process_papers(args.csv_file, args.output_dir)

if __name__ == "__main__":
    main()