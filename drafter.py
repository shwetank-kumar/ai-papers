import sqlite3
from datetime import datetime, timedelta
import json
import os
import textwrap

class Draft:
    def __init__(self, db_path: str = 'paper_summaries.db'):
        self.db_path = db_path
        
    def get_date_range(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT MIN(email_date), MAX(email_date)
                    FROM hf_email
                    WHERE downloaded = 1 AND summarized = 1 AND post_generated = 0''')
        start_date, end_date = c.fetchone()
        conn.close()
        return start_date, end_date

    def get_eligible_papers(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT h.arxiv_id, h.title, h.upvotes, h.email_date,
                     s.main_problem, s.approach, s.findings, s.impact, 
                     s.limitations, s.innovations, s.key_figures, s.primary_results, s.tags
                     FROM hf_email h
                     JOIN summaries s ON h.arxiv_id = s.arxiv_id
                     WHERE h.downloaded = 1 AND h.summarized = 1 AND h.post_generated = 0
                     ORDER BY h.upvotes DESC''')
        papers = c.fetchall()
        conn.close()
        return papers


    def create_draft(self, output_file: str):
        papers = self.get_eligible_papers()
        if not papers:
            print("No eligible papers found for drafting.")
            return

        # Get date range from the database
        start_date, end_date = self.get_date_range()
        # start_date = datetime.strptime(start_date, '%d-%m-%Y')
        # end_date = datetime.strptime(end_date, '%d-%m-%Y')
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Collect all top tags
        all_top_tags = []
        for paper in papers:
            tags = json.loads(paper[-1])
            if tags:
                all_top_tags.append(tags[0])

        with open(output_file, 'w') as f:
            # Write YAML frontmatter (unchanged)
            f.write("---\n")
            f.write(f"title: \"🌙 AI Afterhours: Top AI Papers for {start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}\"\n")
            f.write("author: \"Shwetank Kumar\"\n")
            f.write(f"date: \"{end_date.strftime('%b %d, %Y')}\"\n")
            f.write(f"categories: [{', '.join(set(all_top_tags))}]\n")
            f.write("draft: true\n")
            f.write("page-layout: article\n")
            f.write("---\n\n")

            # Write introduction (updated)
            intro_text = f"""Welcome to this week's AI Afterhours! Your weekly digest of most upvoted papers in AI. Below is gist of the results, how they got them, and why you should care. With that, let's dive into the most exciting AI research from {start_date.strftime('%B %d')} to {end_date.strftime('%B %d, %Y')}. \n\n
            
            <iframe src="https://podcasters.spotify.com/pod/show/shwetankkumar/embed" height="200px" width="400px" frameborder="0" scrolling="no"></iframe>

            <iframe src="../../subscribe.html" width="600" height="400" class="newsletter-form"></iframe>   
            """
            f.write(intro_text + "\n\n")
            f.write("# Summaries\n\n")

            for paper in papers:
                arxiv_id, title, upvotes, date, main_problem, approach, findings, impact, limitations, innovations, key_figures, primary_results, tags = paper
                
                # Write paper title with upvotes
                f.write(f"#### {title} ↑{upvotes}\n\n")

                # Write summary sections (unchanged)
                sections = [
                    ('Main Problem', main_problem),
                    ('Approach', approach),
                    ('Findings', findings),
                    ('Impact', impact),
                ]
                for section_title, content in sections:
                    f.write(f"*{section_title}:* {textwrap.fill(content, width=80, subsequent_indent='  ')}\n\n")

                # Write key results (unchanged)
                f.write("*Key Results:*\n")
                for result in json.loads(primary_results):
                    f.write(f"- {textwrap.fill(result, width=80, subsequent_indent='  ')}\n")
                f.write("\n")

                # Write additional sections (if content is available)
                if limitations:
                    f.write(f"*Limitations:* {textwrap.fill(limitations, width=80, subsequent_indent='  ')}\n\n")
                if innovations:
                    f.write(f"*Innovations:* {textwrap.fill(innovations, width=80, subsequent_indent='  ')}\n\n")
                if key_figures:
                    f.write(f"*Key Figures:* {textwrap.fill(key_figures, width=80, subsequent_indent='  ')}\n\n")

                # Write ArXiv link
                paper_url = f"https://arxiv.org/pdf/{arxiv_id}"
                f.write(f"ArXiv: [{arxiv_id}]({paper_url})\n\n")

                # Write tags
                all_tags = json.loads(tags)
                f.write(f"<small>🏷️ {', '.join(all_tags)}</small>\n\n")
                
                f.write("---\n\n")  # Separator between papers

        return [paper[0] for paper in papers]  # Return list of arxiv_ids

    def update_post_generated_status(self, arxiv_ids: list):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany('''UPDATE hf_email SET post_generated = 1 WHERE arxiv_id = ?''', 
                      [(arxiv_id,) for arxiv_id in arxiv_ids])
        conn.commit()
        conn.close()

    def generate_draft(self, output_file: str):
        processed_arxiv_ids = self.create_draft(output_file)
        if processed_arxiv_ids:
            self.update_post_generated_status(processed_arxiv_ids)
            print(f"Created draft for {len(processed_arxiv_ids)} papers in {output_file}")
        else:
            print("No drafts were created.")

# Usage example
if __name__ == "__main__":
    draft_generator = Draft()
    draft_generator.generate_draft("ai_paper_summaries.qmd")