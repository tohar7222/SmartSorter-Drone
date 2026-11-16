import os

# שימי כאן את הנתיב המדויק לתיקייה שיצרת עם 80 הקבצים החדשים
new_labels_dir = r"C:\טוהר בלולו\פרויקטם\FIX_LABELS" # תשני את זה לנתיב שלך!

count = 0
for filename in os.listdir(new_labels_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(new_labels_dir, filename)
        
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                old_class = int(parts[0])
                # הקסם: מוסיפים 1 לכל תגית. 0 הופך ל-1 (פתוחה), ו-1 הופך ל-2 (סגורה)
                new_class = old_class + 1 
                parts[0] = str(new_class)
                new_lines.append(" ".join(parts) + "\n")
                
        with open(filepath, 'w') as file:
            file.writelines(new_lines)
        count += 1

print(f"תוקנו בהצלחה {count} קבצי תיוג חדשים! כולם הועלו במספר אחד למעלה.")