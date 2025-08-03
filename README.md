# Expense Management API Service

## Overview

This API service provides expense classification and entity recognition for Arabic text. It uses machine learning models to categorize expenses and extract relevant entities such as amounts, dates, and currencies.

## API Endpoints

The service provides two main endpoints:

### 1. Process Expense (Full Analysis)

```
POST /process_expense/
```

This endpoint performs complete analysis of expense text, including:
- Classification of expense category
- Entity recognition (amounts, dates, currencies, etc.)
- Post-processing of entities to standardize formats

#### Request Format

```json
{
  "text": "صرفت خمسين الف ريال على الايجار"
}
```

#### Response Format

```json
{
  "category": "Housing",
  "entities": [
    {
      "entity": "AMOUNT",
      "word": "50000"
    },
    {
      "entity": "CURRENCY",
      "word": "YER"
    }
  ]
}
```

### 2. Classify Expense (Category Only)

```
POST /classify
```

This endpoint only performs expense category classification without entity recognition.

#### Request Format

```json
{
  "text": "صرفت خمسين الف ريال على الايجار"
}
```

#### Response Format

```json
{
  "category": "Housing"
}
```

## Integration Guide

### Flutter Integration

#### 1. Add HTTP Package

Add the http package to your `pubspec.yaml`:

```yaml
dependencies:
  http: ^1.1.0
```

Run `flutter pub get` to install the package.

#### 2. Create API Service

Create a service class to handle API calls:

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class ExpenseApiService {
  final String baseUrl;
  
  ExpenseApiService({required this.baseUrl});
  
  Future<Map<String, dynamic>> processExpense(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/process_expense/'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'text': text}),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to process expense: ${response.body}');
    }
  }
  
  Future<Map<String, dynamic>> classifyExpense(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/classify'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'text': text}),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to classify expense: ${response.body}');
    }
  }
}
```

#### 3. Using the API Service

```dart
final apiService = ExpenseApiService(baseUrl: 'http://your-api-server:8000');

// Process full expense
try {
  final result = await apiService.processExpense('صرفت خمسين الف ريال على الايجار');
  print('Category: ${result['category']}');
  print('Entities: ${result['entities']}');
} catch (e) {
  print('Error: $e');
}

// Classify expense only
try {
  final result = await apiService.classifyExpense('صرفت خمسين الف ريال على الايجار');
  print('Category: ${result['category']}');
} catch (e) {
  print('Error: $e');
}
```

### Web Integration (JavaScript)

#### Using Fetch API

```javascript
// Process full expense
async function processExpense(text) {
  try {
    const response = await fetch('http://your-api-server:8000/process_expense/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error processing expense:', error);
    throw error;
  }
}

// Classify expense only
async function classifyExpense(text) {
  try {
    const response = await fetch('http://your-api-server:8000/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error classifying expense:', error);
    throw error;
  }
}

// Example usage
processExpense('صرفت خمسين الف ريال على الايجار')
  .then(result => {
    console.log('Category:', result.category);
    console.log('Entities:', result.entities);
  })
  .catch(error => {
    console.error('Failed to process expense:', error);
  });
```

## Running the API Server

To run the API server locally:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access the auto-generated API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Understanding and Handling Entities

Entities are key pieces of information extracted from expense text. The API identifies and standardizes these entities to make them easier to use in your application.

### Entity Types

1. **AMOUNT**: Monetary amounts
   - Input: Arabic text like "خمسين الف" (fifty thousand)
   - Output: Standardized numerical format (e.g., "50000")
   - Usage: Can be directly used for calculations or display

2. **DATE**: Date expressions
   - Input: Arabic text like "العشرين من شهر رمضان" 
   - Output: Standardized to YYYY-MM-DD format when possible
   - Usage: Can be used for filtering, sorting, or calendar display

3. **CURRENCY**: Currency mentions
   - Input: Arabic text like "ريال" (riyal)
   - Output: Standardized to ISO codes (e.g., "YER" for Yemeni Riyal, "USD" for US Dollar)
   - Usage: Can be used for currency conversion or display

### Handling Entities in Your Application

#### Flutter Example

```dart
// Processing entities from API response
void processEntities(List<dynamic> entities) {
  String? amount;
  String? currency;
  String? date;
  
  for (var entity in entities) {
    switch(entity['entity']) {
      case 'AMOUNT':
        amount = entity['word'];
        break;
      case 'CURRENCY':
        currency = entity['word'];
        break;
      case 'DATE':
        date = entity['word'];
        break;
    }
  }
  
  // Now you can use these values in your app
  if (amount != null && currency != null) {
    print('Expense amount: $amount $currency');
    // Convert to double for calculations
    double? numericAmount = double.tryParse(amount);
    if (numericAmount != null) {
      // Perform calculations
      double inUSD = convertCurrency(numericAmount, currency, 'USD');
      print('Amount in USD: $inUSD');
    }
  }
  
  if (date != null) {
    print('Expense date: $date');
    // Parse date for date-based operations
    DateTime? expenseDate = DateTime.tryParse(date);
    if (expenseDate != null) {
      // Use for date filtering, etc.
    }
  }
}
```

#### JavaScript Example

```javascript
// Processing entities from API response
function processEntities(entities) {
  let amount = null;
  let currency = null;
  let date = null;
  
  entities.forEach(entity => {
    switch(entity.entity) {
      case 'AMOUNT':
        amount = entity.word;
        break;
      case 'CURRENCY':
        currency = entity.word;
        break;
      case 'DATE':
        date = entity.word;
        break;
    }
  });
  
  // Now you can use these values in your app
  if (amount && currency) {
    console.log(`Expense amount: ${amount} ${currency}`);
    // Convert to number for calculations
    const numericAmount = parseFloat(amount);
    if (!isNaN(numericAmount)) {
      // Perform calculations
      const inUSD = convertCurrency(numericAmount, currency, 'USD');
      console.log(`Amount in USD: ${inUSD}`);
    }
  }
  
  if (date) {
    console.log(`Expense date: ${date}`);
    // Parse date for date-based operations
    const expenseDate = new Date(date);
    if (!isNaN(expenseDate.getTime())) {
      // Use for date filtering, etc.
    }
  }
}
```

## Notes

- The API expects Arabic text input
- For production deployment, update the CORS settings in `main.py` to restrict access to specific origins