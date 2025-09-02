
import os
import sqlite3
import json
import argparse
from datetime import date, datetime, timedelta
from typing import Optional
import textwrap
import sys

try:
    from openai import OpenAI
    OPENAI_NEW = True
except Exception:
    try:
        import openai
        OPENAI_NEW = False
    except Exception:
        OPENAI_NEW = False

DB_PATH = os.path.expanduser("~/.vocab_trainer/vocab.db")

class VocabDB:
    def __init__(self, path=DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY,
            word TEXT UNIQUE,
            meaning TEXT,
            repetition INTEGER DEFAULT 0,
            easiness REAL DEFAULT 2.5,
            interval INTEGER DEFAULT 1,
            priority INTEGER DEFAULT 0,
            next_review TEXT,
            last_review TEXT,
            examples TEXT
        )
        ''')
        self.conn.commit()

    def add_word(self, word: str, meaning: str):
        cur = self.conn.cursor()
        today = date.today().isoformat()
        try:
            cur.execute('''INSERT INTO words (word, meaning, next_review) VALUES (?,?,?)''', (word, meaning, today))
            self.conn.commit()
            print(f"کلمه '{word}' اضافه شد.")
        except sqlite3.IntegrityError:
            print(f"کلمه '{word}' قبلا وجود دارد. برای ویرایش از گزینهٔ list استفاده کنید.")

    def list_words(self):
        cur = self.conn.cursor()
        for row in cur.execute('SELECT id, word, meaning, repetition, interval, priority, next_review FROM words ORDER BY priority DESC, next_review ASC'):
            print(row)

    def get_due_words(self, limit=10):
        cur = self.conn.cursor()
        today = date.today().isoformat()
        rows = cur.execute('''SELECT id, word, meaning, repetition, easiness, interval, priority, next_review, examples
                              FROM words
                              WHERE next_review <= ?
                              ORDER BY priority DESC, next_review ASC
                              LIMIT ?''', (today, limit)).fetchall()
        return rows

    def update_after_review(self, word_id: int, correct: bool, new_values: dict):
        # new_values can contain repetition, easiness, interval, next_review, priority, last_review, examples
        cur = self.conn.cursor()
        set_clause = ', '.join([f"{k} = ?" for k in new_values.keys()])
        params = list(new_values.values()) + [word_id]
        cur.execute(f"UPDATE words SET {set_clause} WHERE id = ?", params)
        self.conn.commit()

    def get_word_by_id(self, word_id: int):
        cur = self.conn.cursor()
        return cur.execute('SELECT id, word, meaning, repetition, easiness, interval, priority, next_review, examples FROM words WHERE id = ?', (word_id,)).fetchone()

    def set_examples(self, word_id: int, examples: dict):
        self.update_after_review(word_id, False, { 'examples': json.dumps(examples) })

def sm2_update(repetition:int, easiness:float, interval:int, quality:int):
    """
    quality: 0..5 (5 = perfect)
    Returns (new_repetition, new_easiness, new_interval)
    """
    assert 0 <= quality <= 5
    if quality < 3:
        repetition = 0
        interval = 1
    else:
        repetition += 1
        if repetition == 1:
            interval = 1
        elif repetition == 2:
            interval = 6
        else:
            interval = max(1, int(round(interval * easiness)))
    # update easiness factor
    easiness = max(1.3, easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    return repetition, easiness, interval

def make_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("اخطار: متغیر محیطی OPENAI_API_KEY پیدا نشد. درخواست به ChatGPT فعال نخواهد بود.")
    try:
        if OPENAI_NEW:
            client = OpenAI()
            return ('new', client)
        else:
            openai.api_key = api_key
            return ('old', openai)
    except Exception as e:
        print("خطا هنگام ساخت کلاینت OpenAI:", e)
        return (None, None)


def fetch_examples_with_chatgpt(word: str, meaning: str, n_examples: int = 5):
    mode, client = make_openai_client()
    if client is None:
        return None

    prompt = textwrap.dedent(f"""
    You are a helpful language teacher. Provide {n_examples} natural example sentences for the English word "{word}" with the meaning: "{meaning}".
    For each example, mark the sentence, a short translation (or paraphrase) in Persian, and highlight the meaning's usage note if relevant.
    Output JSON with keys: word, meaning, examples -> list of {{sentence, paraphrase_persian, note}}.
    Keep output strictly as JSON.
    """)

    try:
        if mode == 'new':
            # new client uses chat.completions.create
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system", "content":"You are a helpful language teacher who responds in JSON when requested."},
                    {"role":"user", "content":prompt}
                ],
                temperature=0.7,
                max_tokens=800,
            )
            text = resp.choices[0].message['content'] if hasattr(resp.choices[0], 'message') or True else resp.choices[0].message['content']
            # NOTE: different SDK versions return slightly different shapes; be permissive
            # Try to get text
            try:
                text = resp.choices[0].message['content']
            except Exception:
                try:
                    text = resp.choices[0].text
                except Exception:
                    text = str(resp)
        else:
            # old openai package
            resp = client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system", "content":"You are a helpful language teacher who responds in JSON when requested."},
                    {"role":"user", "content":prompt}
                ],
                temperature=0.7,
                max_tokens=800,
            )
            text = resp.choices[0].message['content']

        # attempt to find JSON blob
        start = text.find('{')
        if start != -1:
            json_text = text[start:]
            try:
                parsed = json.loads(json_text)
                return parsed
            except Exception:
                return {"word": word, "meaning": meaning, "raw": text}
        else:
            return {"word": word, "meaning": meaning, "raw": text}

    except Exception as e:
        print("خطا هنگام تماس با OpenAI:", e)
        return None

def practice_session(db: VocabDB, n=10):
    due = db.get_due_words(limit=n)
    if not due:
        print("هیچ کلمهٔ آمادهٔ مرور امروز نیست. از add کلمه جدید اضافه کنید یا next_review ها را بررسی کنید.")
        return

    client_mode, _ = make_openai_client()

    for row in due:
        wid, word, meaning, repetition, easiness, interval, priority, next_review, examples = row
        print('\n' + '='*40)
        print(f"کلمه: {word}")
        print(f"معنی (که شما وارد کرده‌اید): {meaning}")
        ans = input("معنی/ترجمه را وارد کنید (یا Enter برای نمایش معنی): ").strip()
        if ans == "":
            print("معنی: ", meaning)
            quality = 0
        else: 
            if ans.lower() in meaning.lower() or meaning.lower() in ans.lower():
                print("درست! ✅")
                quality = 5
            else:
                print("اشتباه ❌ — معنی درست:", meaning)
                quality = 2

        new_rep, new_eas, new_interval = sm2_update(repetition, easiness, interval, quality)
        next_date = (date.today() + timedelta(days=new_interval)).isoformat()
        new_priority = priority
        if quality < 3:
            new_priority = priority + 2
        else:
            new_priority = max(0, priority - 1)

        db.update_after_review(wid, quality>=3, {
            'repetition': new_rep,
            'easiness': new_eas,
            'interval': new_interval,
            'next_review': next_date,
            'last_review': date.today().isoformat(),
            'priority': new_priority
        })

        if quality < 3:
            print("درخواست مثال‌ها و جملات توسط ChatGPT برای کمک به یادگیری...")
            ex = fetch_examples_with_chatgpt(word, meaning)
            if ex:
                db.set_examples(wid, ex)
                print("مثال‌ها ذخیره شدند. چند مثال نمونه:")
                try:
                    for e in ex.get('examples', [])[:3]:
                        print('-', e.get('sentence'))
                        print('  ', e.get('paraphrase_persian'))
                except Exception:
                    print('خروجی ChatGPT (خام):')
                    print(ex)
            else:
                print("دریافت مثال از ChatGPT موفق نبود.")

    print('\nمرور امروز تمام شد.')

def main():
    parser = argparse.ArgumentParser(description='Vocab trainer — یادگیری کلمات با Python و ChatGPT')
    sub = parser.add_subparsers(dest='cmd')

    p_add = sub.add_parser('add')
    p_add.add_argument('word')
    p_add.add_argument('meaning')

    p_practice = sub.add_parser('practice')
    p_practice.add_argument('--n', type=int, default=10)

    p_list = sub.add_parser('list')

    args = parser.parse_args()
    db = VocabDB()

    if args.cmd == 'add':
        db.add_word(args.word, args.meaning)
    elif args.cmd == 'practice':
        practice_session(db, n=args.n)
    elif args.cmd == 'list':
        db.list_words()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
