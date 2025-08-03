import os
import sys
from post_processing import EntityPostProcessor

def test_amount_parsing():
    processor = EntityPostProcessor()
    
    # Test cases
    test_cases = [
        "ست عشر الف وتسعميه",  # 16,900
        "خمسين الف",           # 50,000
        "ثلاثة الاف وخمس مائة", # 3,500
        "عشرين الف ومائتين",    # 20,200
        "احد وعشرين الف وثمان ميه", # 21,800
        "اثنين وعشرين الف وثمان ميه", # 22,800
        "الف وميه",            # 1,100
        "الفين ونص"            # 2,500
    ]
    
    print("Testing amount parsing...")
    print("-" * 50)
    
    for text in test_cases:
        result = processor._standardize_amount(text)
        print(f"Input: {text}")
        print(f"Result: {result}")
        print("-" * 50)

if __name__ == "__main__":
    test_amount_parsing()