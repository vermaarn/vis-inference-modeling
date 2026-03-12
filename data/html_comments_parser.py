import json
from bs4 import BeautifulSoup

def parse_html_to_json(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []

    # Find all top-level comment containers
    # We use the data-testid which is less likely to change than randomized CSS classes
    comments = soup.find_all("div", {"data-testid": "comment-container"})

    for comment in comments:
        # Extract basic info using semantic markers or data-testids
        user_header = comment.find("div", {"data-testid": "user-header"})
        if not user_header:
            continue
            
        # Name is usually the first div inside the user header
        # name = user_header.find("div").get_text(strip=True)
        # get the first a tag inside the user header
        name = comment.find("a").get_text(strip=True)

        # remove everything after the first encounte of 'commented'
        name = name.split('commented')[0].strip()
        
        # Subtitle usually contains the location
        subtitle_span = user_header.find("span", {"data-testid": "user-header-subtitle"})
        subtitle_text = subtitle_span.get_text(strip=True) if subtitle_span else ""
        # Clean up location (strip the trailing dot/separator)
        location = subtitle_text.split(' ·')[0].strip() if ' ·' in subtitle_text else subtitle_text
        
        # Date is marked with a specific test id
        date_span = comment.find("span", {"data-testid": "todays-date"})
        date = date_span.get_text(strip=True) if date_span else ""
        
        # Comment text is the paragraph inside the content area
        # We find the 'p' tag that isn't inside a nested reply list
        comment_text_element = comment.find("p")
        comment_text = comment_text_element.get_text(strip=True) if comment_text_element else ""

        entry = {
            "name": name,
            "location": location,
            "date posted": date,
            "comment info": comment_text,
            "replies": []
        }

        # Check for nested replies
        # Replies are contained in a div with a specific threading test id
        replies_list = comment.find("div", {"data-testid": "reply-list-threading"})
        
        if replies_list:
            # Find individual reply containers
            reply_containers = replies_list.find_all("div", {"data-testid": "reply-comment-container"})
            
            for reply in reply_containers:
                r_header = reply.find("div", {"data-testid": "user-header"})
                r_name = r_header.find("div").get_text(strip=True)
                
                r_subtitle = reply.find("span", {"data-testid": "user-header-subtitle"}).get_text(strip=True)
                r_loc = r_subtitle.split(' ·')[0].strip() if ' ·' in r_subtitle else r_subtitle
                
                r_date = reply.find("span", {"data-testid": "todays-date"}).get_text(strip=True)
                r_text = reply.find("p").get_text(strip=True)

                reply_obj = {
                    "name": r_name,
                    "location": r_loc,
                    "date posted": r_date,
                    "comment info": r_text,
                    "reply to": name # The parent author
                }
                entry["replies"].append(reply_obj)

        results.append(entry)

    return results

if __name__ == "__main__":
    import os
    
    for file in os.listdir("data/article_comments_html"):
        if file.endswith(".html"):
            with open(f"data/article_comments_html/{file}", 'r', encoding='utf-8') as f:
                html_content = f.read()
            result = parse_html_to_json(html_content)
            with open(f"data/comment_data/{file.split('.')[0]}.json", "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Saved {file.split('.')[0]}.json")
        else:
            print(f"Skipping {file} because it is not a HTML file")