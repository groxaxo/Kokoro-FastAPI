# Text Sanitization Strategies Guide / Guía de Estrategias de Sanitización de Texto

> **English** | [Español](#español)

---

## English

### Overview

This document outlines comprehensive text sanitization and normalization strategies used in the Kokoro-FastAPI text-to-speech system. These strategies ensure that input text is properly processed, cleaned, and transformed into a format that produces high-quality, natural-sounding speech output.

The sanitization system is designed to handle various types of input including:
- URLs and email addresses
- Numbers, money, and measurements
- Time and phone numbers
- Special characters and symbols
- Multi-language text (with focus on English and Spanish)

### Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Input Validation](#input-validation)
3. [Text Normalization](#text-normalization)
4. [Character Sanitization](#character-sanitization)
5. [Configuration Options](#configuration-options)
6. [Practical Examples](#practical-examples)
7. [Adapting to Your Project](#adapting-to-your-project)

---

### Architecture Overview

The sanitization pipeline consists of three main stages:

```
Raw Text Input
    ↓
[1. Input Validation] → Pydantic schema validation
    ↓
[2. Text Normalization] → Transform special formats
    ↓
[3. Character Sanitization] → Clean special characters
    ↓
Processed Text Output
```

#### Processing Flow

1. **Request Reception**: Text arrives via OpenAI-compatible API endpoint
2. **Schema Validation**: Pydantic models validate structure and types
3. **Text Normalization**: Conditional transformations based on configuration
4. **Phonemization**: Text converted to phonemes for TTS processing
5. **Tokenization**: Phonemes converted to model input tokens

---

### Input Validation

#### Schema-Based Validation

**Location**: `api/src/structures/schemas.py`

All incoming requests are validated using Pydantic models to ensure type safety and structure correctness.

```python
class OpenAISpeechRequest(BaseModel):
    """Request validation schema"""
    
    model: str = Field(
        default="kokoro",
        description="Model identifier"
    )
    
    input: str = Field(
        ..., 
        description="Text to process (required)"
    )
    
    voice: str = Field(
        default="af_heart",
        description="Voice identifier"
    )
    
    speed: float = Field(
        default=1.0,
        ge=0.25,  # Minimum value
        le=4.0,   # Maximum value
        description="Speed multiplier"
    )
```

**Key Validation Points**:

1. **Required Fields**: The `input` field is mandatory (denoted by `...`)
2. **Type Enforcement**: All fields have strict type checking
3. **Range Constraints**: Numeric fields have minimum/maximum bounds
4. **Default Values**: Sensible defaults prevent missing value errors

**Security Considerations**:

- **Type Safety**: Prevents injection of non-string data into text processing
- **Boundary Protection**: Speed limits prevent resource exhaustion
- **Default Fallbacks**: Reduce attack surface from missing parameters

---

### Text Normalization

**Location**: `api/src/services/text_processing/normalizer.py`

Text normalization transforms various formats into speakable text. Each normalization type can be enabled/disabled independently.

#### 1. URL Normalization

Converts URLs into pronounceable format.

**Strategy**:
- Extract protocol (https, http, www)
- Split domain parts with "dot"
- Convert special characters to words
- Handle paths and query parameters

**Examples**:

```python
# Input:  "Visit https://example.com/path"
# Output: "Visit https example dot com slash path"

# Input:  "Check www.test.org?q=search"
# Output: "Check www test dot org question-mark q equals search"

# Input:  "API at localhost:8080/api"
# Output: "API at localhost colon eighty eighty slash api"
```

**Implementation Pattern**:

```python
URL_PATTERN = re.compile(
    r"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+\.(?:com|org|...))",
    re.IGNORECASE
)

def handle_url(match):
    url = match.group(0).strip()
    
    # Handle protocol
    url = re.sub(r"^https?://", 
                 lambda m: "https " if "https" in m.group() else "http ",
                 url)
    
    # Split into domain and path
    parts = url.split("/", 1)
    domain = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    
    # Handle domain dots
    domain = domain.replace(".", " dot ")
    
    # Reconstruct and handle special characters
    url = f"{domain} slash {path}" if path else domain
    url = url.replace("?", " question-mark ")
    url = url.replace("=", " equals ")
    
    return url
```

**Configuration**:
```python
normalization_options = NormalizationOptions(
    url_normalization=True  # Enable URL processing
)
```

---

#### 2. Email Normalization

Converts email addresses to speakable format.

**Strategy**:
- Split on `@` symbol
- Replace dots in domain with "dot"
- Keep username readable

**Examples**:

```python
# Input:  "Contact user@example.com"
# Output: "Contact user at example dot com"

# Input:  "Email test.user@site.org"
# Output: "Email test dot user at site dot org"
```

**Implementation Pattern**:

```python
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b",
    re.IGNORECASE
)

def handle_email(match):
    email = match.group(0)
    user, domain = email.split("@")
    domain = domain.replace(".", " dot ")
    return f"{user} at {domain}"
```

**Configuration**:
```python
normalization_options = NormalizationOptions(
    email_normalization=True  # Enable email processing
)
```

---

#### 3. Number Normalization

Converts numeric values to written words.

**Strategy**:
- Parse integer and decimal parts
- Handle negative numbers
- Support abbreviations (k, m, b, t)
- Special handling for 4-digit years

**Examples**:

```python
# Input:  "I have 1035 items"
# Output: "I have one thousand and thirty-five items"

# Input:  "Population: 7.5m people"
# Output: "Population: seven point five million people"

# Input:  "In 1998 we started"
# Output: "In nineteen ninety-eight we started"

# Input:  "Temperature: -5 degrees"
# Output: "Temperature: minus five degrees"
```

**Implementation Pattern**:

```python
NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand|...)*)\b",
    re.IGNORECASE
)

def handle_numbers(match):
    number = float(match.group(2))
    
    if match.group(1) == "-":
        number *= -1
    
    multiplier = translate_multiplier(match.group(3))
    
    # Special handling for 4-digit years
    if (number % 1 == 0 and len(str(number)) == 4 
        and number > 1500):
        return split_four_digit(number)
    
    return f"{number_to_words(number)}{multiplier}"
```

**Decimal Handling**:

```python
# Input:  "Value is 56.789"
# Output: "Value is fifty-six point seven eight nine"

def handle_decimal(match):
    integer, decimal = match.group().split(".")
    return f"{integer} point {' '.join(decimal)}"
```

---

#### 4. Money Normalization

Converts monetary values to spoken format.

**Strategy**:
- Identify currency symbol ($, £, €)
- Handle whole and decimal amounts
- Support multipliers (k, m, b)
- Proper pluralization

**Examples**:

```python
# Input:  "It costs $50.30"
# Output: "It costs fifty dollars and thirty cents"

# Input:  "Price: €30.2"
# Output: "Price: thirty euros and twenty cents"

# Input:  "Lost $5.3k"
# Output: "Lost five point three thousand dollars"

# Input:  "Negative -$100"
# Output: "Negative minus one hundred dollars"
```

**Implementation Pattern**:

```python
MONEY_PATTERN = re.compile(
    r"(-?)([$£€])(\d+(?:\.\d+)?)((?: thousand|...)*)\b",
    re.IGNORECASE
)

MONEY_UNITS = {
    "$": ("dollar", "cent"),
    "£": ("pound", "pence"),
    "€": ("euro", "cent")
}

def handle_money(match):
    bill, coin = MONEY_UNITS[match.group(2)]
    number = float(match.group(3))
    
    if match.group(1) == "-":
        number *= -1
    
    if number % 1 == 0:
        # Whole amount
        return f"{number_to_words(number)} {plural(bill, number)}"
    else:
        # With cents
        dollars = int(number)
        cents = int(str(number).split(".")[-1].ljust(2, "0"))
        return f"{number_to_words(dollars)} {plural(bill)} and {number_to_words(cents)} {plural(coin, cents)}"
```

---

#### 5. Unit Normalization

Converts measurements with units to spoken form.

**Strategy**:
- Detect number + unit pattern
- Expand unit abbreviations
- Handle pluralization
- Support SI and imperial units

**Examples**:

```python
# Input:  "Distance: 5km"
# Output: "Distance: five kilometers"

# Input:  "Speed: 60mph"
# Output: "Speed: sixty miles per hour"

# Input:  "Storage: 128GB"
# Output: "Storage: one hundred and twenty-eight gigabytes"
```

**Supported Units**:

```python
VALID_UNITS = {
    # Length
    "m": "meter", "cm": "centimeter", "km": "kilometer",
    "ft": "foot", "mi": "mile",
    
    # Mass
    "g": "gram", "kg": "kilogram",
    
    # Time
    "s": "second", "min": "minutes", "h": "hour",
    
    # Data
    "b": "bit", "kb": "kilobit", "mb": "megabit", 
    "gb": "gigabit", "tb": "terabit",
    
    # Speed
    "mph": "mile per hour", "kph": "kilometer per hour",
    
    # Temperature
    "°c": "degree celsius", "°f": "degree fahrenheit",
    
    # And many more...
}
```

**Implementation Pattern**:

```python
UNIT_PATTERN = re.compile(
    r"(([+-]?)(\d+)(\.\d+)?)\s*(" + 
    "|".join(VALID_UNITS.keys()) + 
    r")(?=\W|\b)",
    re.IGNORECASE
)

def handle_units(match):
    number = match.group(1).strip()
    unit = match.group(6).strip().lower()
    
    if unit in VALID_UNITS:
        unit_word = VALID_UNITS[unit]
        
        # Handle bit vs byte (B vs b)
        if unit_word.endswith("bit") and unit[-1] == "B":
            unit_word = unit_word[:-3] + "byte"
        
        # Pluralize
        unit_word = plural(unit_word, number)
    
    return f"{number} {unit_word}"
```

**Configuration**:
```python
normalization_options = NormalizationOptions(
    unit_normalization=True  # Enable unit processing
)
```

---

#### 6. Time Normalization

Converts time expressions to spoken format.

**Strategy**:
- Parse HH:MM or HH:MM:SS format
- Handle AM/PM indicators
- Special handling for o'clock

**Examples**:

```python
# Input:  "Meeting at 10:35 pm"
# Output: "Meeting at ten thirty-five pm"

# Input:  "Departure 5:03 am"
# Output: "Departure five oh three am"

# Input:  "It's 3:00"
# Output: "It's three o'clock"

# Input:  "Time: 13:42:05"
# Output: "Time: thirteen forty-two and five seconds"
```

**Implementation Pattern**:

```python
TIME_PATTERN = re.compile(
    r"([0-9]{1,2}:[0-9]{2}(:[0-9]{2})?)(\s?(pm|am)\b)?",
    re.IGNORECASE
)

def handle_time(match):
    parts = match.group(1).split(":")
    
    result = []
    
    # Hours
    result.append(number_to_words(parts[0].strip()))
    
    # Minutes
    minutes = int(parts[1])
    if minutes < 10 and minutes != 0:
        result.append(f"oh {number_to_words(minutes)}")
    elif minutes == 0:
        result.append("o'clock")
    else:
        result.append(number_to_words(minutes))
    
    # Seconds (if present)
    if len(parts) > 2:
        seconds = int(parts[2])
        result.append(f"and {number_to_words(seconds)} seconds")
    
    # AM/PM
    if match.group(3):
        result.append(match.group(3).strip())
    
    return " ".join(result)
```

**Configuration**:
```python
# Time normalization is always enabled when normalize=True
normalization_options = NormalizationOptions(
    normalize=True
)
```

---

#### 7. Phone Number Normalization

Converts phone numbers to spoken digit groups.

**Strategy**:
- Parse country code, area code, prefix, line number
- Group digits for natural speech
- Handle optional country code

**Examples**:

```python
# Input:  "Call (555) 123-4567"
# Output: "Call five five five, one two three, four five six seven"

# Input:  "+1 (800) 555-0123"
# Output: "+one, eight zero zero, five five five, zero one two three"
```

**Implementation Pattern**:

```python
PHONE_PATTERN = re.compile(
    r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})"
)

def handle_phone_number(match):
    parts = []
    
    # Country code
    if match.group(1):
        parts.append(number_to_words(match.group(1), group=1))
    
    # Area code
    area = match.group(3).replace("(", "").replace(")", "")
    parts.append(number_to_words(area, group=1))
    
    # Prefix
    parts.append(number_to_words(match.group(4), group=1))
    
    # Line number
    parts.append(number_to_words(match.group(5), group=1))
    
    return ", ".join(parts)
```

**Configuration**:
```python
normalization_options = NormalizationOptions(
    phone_normalization=True  # Enable phone processing
)
```

---

### Character Sanitization

**Location**: `api/src/services/text_processing/normalizer.py`

After normalization, remaining special characters are handled.

#### Symbol Replacement

**Strategy**: Replace symbols with spoken equivalents or remove them.

```python
SYMBOL_REPLACEMENTS = {
    '~': ' ',
    '@': ' at ',
    '#': ' number ',
    '$': ' dollar ',
    '%': ' percent ',
    '^': ' ',
    '&': ' and ',
    '*': ' ',
    '_': ' ',
    '|': ' ',
    '\\': ' ',
    '/': ' slash ',
    '=': ' equals ',
    '+': ' plus ',
}
```

**Examples**:

```python
# Input:  "Buy products @ store & online"
# Output: "Buy products at store and online"

# Input:  "Discount: 20% off"
# Output: "Discount: twenty percent off"

# Input:  "Formula: x + y = z"
# Output: "Formula: x plus y equals z"
```

**Configuration**:
```python
normalization_options = NormalizationOptions(
    replace_remaining_symbols=True  # Enable symbol replacement
)
```

---

#### Quote and Bracket Normalization

**Strategy**: Standardize various quote styles to basic ASCII quotes.

```python
# Unicode quotes to ASCII
text = text.replace(chr(8216), "'")  # ' → '
text = text.replace(chr(8217), "'")  # ' → '
text = text.replace(chr(8220), '"')  # " → "
text = text.replace(chr(8221), '"')  # " → "

# Guillemets to quotes
text = text.replace("«", '"')
text = text.replace("»", '"')
```

**Examples**:

```python
# Input:  "He said 'hello'"
# Output: "He said 'hello'"

# Input:  "Book title: «Example»"
# Output: "Book title: "Example""
```

---

#### CJK Punctuation

**Strategy**: Convert Chinese/Japanese punctuation to Western equivalents.

```python
# CJK to Western punctuation
replacements = {
    '、': ', ',  # Enumeration comma
    '。': '. ',  # Period
    '！': '! ',  # Exclamation
    '，': ', ',  # Comma
    '：': ': ',  # Colon
    '；': '; ',  # Semicolon
    '？': '? ',  # Question mark
    '–': '- ',   # En dash
}
```

---

#### Whitespace Normalization

**Strategy**: Clean and standardize all whitespace.

```python
# Remove non-standard whitespace (tabs, etc.)
text = re.sub(r"[^\S \n]", " ", text)

# Collapse multiple spaces
text = re.sub(r"  +", " ", text)

# Remove spaces in blank lines
text = re.sub(r"(?<=\n) +(?=\n)", "", text)

# Convert newlines to spaces
text = text.replace('\n', ' ')
text = text.replace('\r', ' ')
```

---

#### Abbreviation Expansion

**Strategy**: Expand common abbreviations for better pronunciation.

```python
# Titles
text = re.sub(r"\bDr\.(?= [A-Z])", "Doctor", text)
text = re.sub(r"\bMr\.(?= [A-Z])", "Mister", text)
text = re.sub(r"\bMs\.(?= [A-Z])", "Miss", text)
text = re.sub(r"\bMrs\.(?= [A-Z])", "Mrs", text)
text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
```

**Examples**:

```python
# Input:  "Dr. Smith is here"
# Output: "Doctor Smith is here"

# Input:  "Mr. Johnson, Ms. Lee, etc."
# Output: "Mister Johnson, Miss Lee, etc"
```

---

### Configuration Options

All sanitization features can be configured independently using the `NormalizationOptions` schema.

#### Complete Configuration Schema

```python
class NormalizationOptions(BaseModel):
    """Sanitization configuration options"""
    
    normalize: bool = Field(
        default=True,
        description="Master switch for all normalization"
    )
    
    url_normalization: bool = Field(
        default=True,
        description="Convert URLs to speakable format"
    )
    
    email_normalization: bool = Field(
        default=True,
        description="Convert emails to speakable format"
    )
    
    unit_normalization: bool = Field(
        default=False,
        description="Convert units (10KB → ten kilobytes)"
    )
    
    phone_normalization: bool = Field(
        default=True,
        description="Convert phone numbers to spoken digits"
    )
    
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Expand (s) to s"
    )
    
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Replace symbols with words"
    )
```

#### Usage Examples

**Full Normalization (Default)**:
```python
options = NormalizationOptions()
# All features enabled
```

**Minimal Normalization**:
```python
options = NormalizationOptions(
    normalize=True,
    url_normalization=False,
    email_normalization=False,
    phone_normalization=False,
    replace_remaining_symbols=False
)
# Only basic normalization
```

**Custom Configuration**:
```python
options = NormalizationOptions(
    normalize=True,
    url_normalization=True,
    unit_normalization=True,  # Enable unit conversion
    email_normalization=False,
    phone_normalization=False
)
# URLs and units only
```

**No Normalization**:
```python
options = NormalizationOptions(
    normalize=False
)
# Bypass all normalization
```

---

### Practical Examples

#### Example 1: Technical Documentation

**Input**:
```
Visit https://api.example.com:8080/docs for documentation. 
Contact support@example.com if you need help. 
The API processes 1.5k requests/sec at 99.9% uptime.
```

**Configuration**:
```python
options = NormalizationOptions(
    normalize=True,
    url_normalization=True,
    email_normalization=True,
    unit_normalization=True
)
```

**Output**:
```
Visit https api dot example dot com colon eighty eighty slash docs for documentation.
Contact support at example dot com if you need help.
The API processes one point five thousand requests slash sec at ninety-nine point nine percent uptime.
```

---

#### Example 2: E-commerce Product

**Input**:
```
Product #12345: Premium Headphones
Price: $249.99 (20% off!)
Weight: 250g
Shipping: Free for orders $50+
Call (800) 555-0199 to order
```

**Configuration**:
```python
options = NormalizationOptions(
    normalize=True,
    unit_normalization=True,
    phone_normalization=True
)
```

**Output**:
```
Product number twelve thousand, three hundred and forty-five: Premium Headphones
Price: two hundred and forty-nine dollars and ninety-nine cents (twenty percent off!)
Weight: two hundred and fifty grams
Shipping: Free for orders fifty dollars plus
Call eight zero zero, five five five, zero one nine nine to order
```

---

#### Example 3: News Article

**Input**:
```
On 1/15/2024 at 10:35 AM, the company announced revenues of $5.3m.
Dr. Smith & Mr. Jones led the presentation @ headquarters.
Stock rose +12.5% to $45.20/share.
```

**Configuration**:
```python
options = NormalizationOptions(
    normalize=True,
    replace_remaining_symbols=True
)
```

**Output**:
```
On one slash fifteen slash twenty twenty-four at ten thirty-five AM, the company announced revenues of five point three million dollars.
Doctor Smith and Mister Jones led the presentation at headquarters.
Stock rose plus twelve point five percent to forty-five dollars and twenty cents slash share.
```

---

### Adapting to Your Project

#### Step 1: Understand Your Requirements

Ask yourself:
- What types of input will you receive?
- Do you need URL/email sanitization?
- Are units and measurements common?
- What language(s) will you support?

#### Step 2: Extract Core Components

The sanitization system has modular components you can adapt:

**Text Normalization Module** (`normalizer.py`):
- Regular expression patterns
- Handler functions for each type
- Configuration-based processing

**Validation Layer** (`schemas.py`):
- Pydantic models for type safety
- Field validators
- Configuration schemas

**Processing Pipeline** (`text_processor.py`):
- Text chunking logic
- Normalization application
- Token generation

#### Step 3: Customize Patterns

Modify regex patterns for your needs:

```python
# Add new currency symbol
MONEY_UNITS = {
    "$": ("dollar", "cent"),
    "£": ("pound", "pence"),
    "€": ("euro", "cent"),
    "¥": ("yen", "sen"),  # Add yen
}

# Add new units
VALID_UNITS = {
    # ... existing units ...
    "bar": "bar",        # Pressure unit
    "psi": "pounds per square inch",
}

# Add new TLDs for URLs
VALID_TLDS = [
    # ... existing TLDs ...
    "app", "dev", "cloud",
]
```

#### Step 4: Create Language-Specific Handlers

For multilingual support:

```python
def normalize_text_with_language(text: str, lang: str, options):
    """Language-aware normalization"""
    
    # Apply language-specific rules
    if lang in ["es", "spanish"]:
        # Spanish-specific handling
        text = handle_spanish_abbreviations(text)
        text = handle_spanish_numbers(text)
    
    elif lang in ["en", "english"]:
        # English-specific handling
        text = normalize_text(text, options)
    
    return text

def handle_spanish_numbers(text: str) -> str:
    """Convert Spanish number formats"""
    # 1.000.000 (European) → 1000000
    text = re.sub(r"(\d+)\.(\d{3})", r"\1\2", text)
    
    # 5,50 (decimal) → 5.50
    text = re.sub(r"(\d+),(\d+)", r"\1.\2", text)
    
    return text
```

#### Step 5: Implement Progressive Enhancement

Start simple, add features as needed:

```python
# Phase 1: Basic validation only
class MinimalRequest(BaseModel):
    text: str
    # No normalization yet

# Phase 2: Add configurable normalization
class BasicRequest(BaseModel):
    text: str
    normalize_urls: bool = True
    normalize_numbers: bool = True

# Phase 3: Full featured (like Kokoro-FastAPI)
class FullRequest(BaseModel):
    input: str
    normalization_options: NormalizationOptions = NormalizationOptions()
```

#### Step 6: Testing Strategy

Create comprehensive test suites:

```python
def test_sanitization_suite():
    """Test various sanitization scenarios"""
    
    test_cases = [
        # URLs
        ("Visit https://example.com", 
         "Visit https example dot com"),
        
        # Money
        ("Costs $50.30", 
         "Costs fifty dollars and thirty cents"),
        
        # Special characters
        ("A & B @ store", 
         "A and B at store"),
        
        # Edge cases
        ("", ""),  # Empty string
        ("   ", ""),  # Whitespace only
        ("123", "one hundred and twenty-three"),
    ]
    
    for input_text, expected in test_cases:
        result = normalize_text(input_text, NormalizationOptions())
        assert result == expected, f"Failed: {input_text}"
```

#### Step 7: Performance Optimization

For production use:

```python
# 1. Pre-compile regex patterns
URL_PATTERN = re.compile(r"...", re.IGNORECASE)  # Compile once

# 2. Use caching for repeated conversions
@lru_cache(maxsize=1000)
def number_to_words_cached(num: int) -> str:
    return inflect_engine.number_to_words(num)

# 3. Lazy initialization of heavy resources
class TextNormalizer:
    _inflect_engine = None
    
    @classmethod
    def get_engine(cls):
        if cls._inflect_engine is None:
            cls._inflect_engine = inflect.engine()
        return cls._inflect_engine
```

#### Step 8: Security Considerations

Implement security best practices:

```python
# 1. Input length limits
class SecureRequest(BaseModel):
    input: str = Field(..., max_length=10000)  # Limit length

# 2. Character whitelist (if needed)
ALLOWED_CHARS = set(string.ascii_letters + string.digits + 
                    string.punctuation + " \n\t")

def sanitize_input(text: str) -> str:
    """Remove potentially dangerous characters"""
    return "".join(c for c in text if c in ALLOWED_CHARS)

# 3. Rate limiting on expensive operations
from functools import wraps
import time

def rate_limit(max_per_minute=60):
    """Decorator to limit function calls"""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls
            calls[:] = [t for t in calls if now - t < 60]
            
            if len(calls) >= max_per_minute:
                raise Exception("Rate limit exceeded")
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

### Best Practices Summary

1. **Validation First**: Always validate input before processing
2. **Fail Safely**: Provide fallbacks for normalization failures
3. **Configuration Over Code**: Make features configurable
4. **Test Extensively**: Cover edge cases and malformed input
5. **Document Behavior**: Explain what each normalization does
6. **Performance Matters**: Pre-compile patterns, cache results
7. **Security Awareness**: Limit input size, sanitize dangerous chars
8. **Modular Design**: Keep normalization functions independent
9. **Language Support**: Handle multilingual text appropriately
10. **Progressive Enhancement**: Start simple, add features incrementally

---

## Español

### Descripción General

Este documento describe las estrategias integrales de sanitización y normalización de texto utilizadas en el sistema de texto a voz Kokoro-FastAPI. Estas estrategias aseguran que el texto de entrada sea procesado, limpiado y transformado adecuadamente en un formato que produce salida de voz natural y de alta calidad.

El sistema de sanitización está diseñado para manejar varios tipos de entrada incluyendo:
- URLs y direcciones de correo electrónico
- Números, dinero y mediciones
- Tiempo y números telefónicos
- Caracteres especiales y símbolos
- Texto multilingüe (con enfoque en inglés y español)

### Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Validación de Entrada](#validación-de-entrada)
3. [Normalización de Texto](#normalización-de-texto)
4. [Sanitización de Caracteres](#sanitización-de-caracteres)
5. [Opciones de Configuración](#opciones-de-configuración)
6. [Ejemplos Prácticos](#ejemplos-prácticos)
7. [Adaptación a Tu Proyecto](#adaptación-a-tu-proyecto)

---

### Arquitectura General

El pipeline de sanitización consiste en tres etapas principales:

```
Entrada de Texto Crudo
    ↓
[1. Validación de Entrada] → Validación de esquema Pydantic
    ↓
[2. Normalización de Texto] → Transformar formatos especiales
    ↓
[3. Sanitización de Caracteres] → Limpiar caracteres especiales
    ↓
Salida de Texto Procesado
```

#### Flujo de Procesamiento

1. **Recepción de Solicitud**: El texto llega vía endpoint API compatible con OpenAI
2. **Validación de Esquema**: Los modelos Pydantic validan estructura y tipos
3. **Normalización de Texto**: Transformaciones condicionales basadas en configuración
4. **Fonemización**: Texto convertido a fonemas para procesamiento TTS
5. **Tokenización**: Fonemas convertidos a tokens de entrada del modelo

---

### Validación de Entrada

#### Validación Basada en Esquema

**Ubicación**: `api/src/structures/schemas.py`

Todas las solicitudes entrantes se validan usando modelos Pydantic para asegurar seguridad de tipos y corrección estructural.

```python
class OpenAISpeechRequest(BaseModel):
    """Esquema de validación de solicitud"""
    
    model: str = Field(
        default="kokoro",
        description="Identificador del modelo"
    )
    
    input: str = Field(
        ..., 
        description="Texto a procesar (requerido)"
    )
    
    voice: str = Field(
        default="af_heart",
        description="Identificador de voz"
    )
    
    speed: float = Field(
        default=1.0,
        ge=0.25,  # Valor mínimo
        le=4.0,   # Valor máximo
        description="Multiplicador de velocidad"
    )
```

**Puntos Clave de Validación**:

1. **Campos Requeridos**: El campo `input` es obligatorio (denotado por `...`)
2. **Imposición de Tipos**: Todos los campos tienen verificación estricta de tipos
3. **Restricciones de Rango**: Los campos numéricos tienen límites mínimo/máximo
4. **Valores Predeterminados**: Los valores predeterminados sensibles previenen errores de valores faltantes

**Consideraciones de Seguridad**:

- **Seguridad de Tipos**: Previene inyección de datos no-string en procesamiento de texto
- **Protección de Límites**: Los límites de velocidad previenen agotamiento de recursos
- **Respaldos Predeterminados**: Reducen la superficie de ataque de parámetros faltantes

---

### Normalización de Texto

**Ubicación**: `api/src/services/text_processing/normalizer.py`

La normalización de texto transforma varios formatos en texto pronunciable. Cada tipo de normalización puede habilitarse/deshabilitarse independientemente.

#### 1. Normalización de URL

Convierte URLs a formato pronunciable.

**Estrategia**:
- Extraer protocolo (https, http, www)
- Dividir partes del dominio con "dot"
- Convertir caracteres especiales a palabras
- Manejar rutas y parámetros de consulta

**Ejemplos**:

```python
# Entrada:  "Visita https://ejemplo.com/ruta"
# Salida:   "Visita https ejemplo dot com slash ruta"

# Entrada:  "Revisa www.prueba.org?q=buscar"
# Salida:   "Revisa www prueba dot org question-mark q equals buscar"

# Entrada:  "API en localhost:8080/api"
# Salida:   "API en localhost colon eighty eighty slash api"
```

**Patrón de Implementación**:

```python
URL_PATTERN = re.compile(
    r"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+\.(?:com|org|...))",
    re.IGNORECASE
)

def handle_url(match):
    url = match.group(0).strip()
    
    # Manejar protocolo
    url = re.sub(r"^https?://", 
                 lambda m: "https " if "https" in m.group() else "http ",
                 url)
    
    # Manejar puntos del dominio
    domain = domain.replace(".", " dot ")
    
    # Manejar caracteres especiales
    url = url.replace("?", " question-mark ")
    url = url.replace("=", " equals ")
    
    return url
```

**Configuración**:
```python
opciones_normalizacion = NormalizationOptions(
    url_normalization=True  # Habilitar procesamiento de URL
)
```

---

#### 2. Normalización de Email

Convierte direcciones de correo electrónico a formato pronunciable.

**Estrategia**:
- Dividir en símbolo `@`
- Reemplazar puntos en dominio con "dot"
- Mantener nombre de usuario legible

**Ejemplos**:

```python
# Entrada:  "Contacto usuario@ejemplo.com"
# Salida:   "Contacto usuario at ejemplo dot com"

# Entrada:  "Email prueba.usuario@sitio.org"
# Salida:   "Email prueba dot usuario at sitio dot org"
```

**Patrón de Implementación**:

```python
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b",
    re.IGNORECASE
)

def handle_email(match):
    email = match.group(0)
    usuario, dominio = email.split("@")
    dominio = dominio.replace(".", " dot ")
    return f"{usuario} at {dominio}"
```

**Configuración**:
```python
opciones_normalizacion = NormalizationOptions(
    email_normalization=True  # Habilitar procesamiento de email
)
```

---

#### 3. Normalización de Números

Convierte valores numéricos a palabras escritas.

**Estrategia**:
- Analizar partes enteras y decimales
- Manejar números negativos
- Soportar abreviaciones (k, m, b, t)
- Manejo especial para años de 4 dígitos

**Ejemplos**:

```python
# Entrada:  "Tengo 1035 artículos"
# Salida:   "Tengo one thousand and thirty-five artículos"

# Entrada:  "Población: 7.5m personas"
# Salida:   "Población: seven point five million personas"

# Entrada:  "En 1998 comenzamos"
# Salida:   "En nineteen ninety-eight comenzamos"

# Entrada:  "Temperatura: -5 grados"
# Salida:   "Temperatura: minus five grados"
```

**Nota**: El sistema actualmente convierte números a palabras en inglés. Para soporte completo en español, se necesitaría implementar un módulo de conversión de números en español. Bibliotecas como [`num2words`](https://pypi.org/project/num2words/) soportan conversión de números a palabras en español:

```python
from num2words import num2words

# Ejemplo de conversión en español
num2words(1035, lang='es')  # 'mil treinta y cinco'
num2words(50.30, lang='es') # 'cincuenta punto tres cero'
```

---

#### 4. Normalización de Dinero

Convierte valores monetarios a formato hablado.

**Estrategia**:
- Identificar símbolo de moneda ($, £, €)
- Manejar cantidades enteras y decimales
- Soportar multiplicadores (k, m, b)
- Pluralización apropiada

**Ejemplos**:

```python
# Entrada:  "Cuesta $50.30"
# Salida:   "Cuesta fifty dollars and thirty cents"

# Entrada:  "Precio: €30.2"
# Salida:   "Precio: thirty euros and twenty cents"

# Entrada:  "Perdió $5.3k"
# Salida:   "Perdió five point three thousand dollars"
```

---

#### 5. Normalización de Unidades

Convierte mediciones con unidades a forma hablada.

**Estrategia**:
- Detectar patrón número + unidad
- Expandir abreviaciones de unidades
- Manejar pluralización
- Soportar unidades SI e imperiales

**Ejemplos**:

```python
# Entrada:  "Distancia: 5km"
# Salida:   "Distancia: five kilometers"

# Entrada:  "Velocidad: 60mph"
# Salida:   "Velocidad: sixty miles per hour"

# Entrada:  "Almacenamiento: 128GB"
# Salida:   "Almacenamiento: one hundred and twenty-eight gigabytes"
```

**Unidades Soportadas**:

```python
VALID_UNITS = {
    # Longitud
    "m": "meter", "cm": "centimeter", "km": "kilometer",
    "ft": "foot", "mi": "mile",
    
    # Masa
    "g": "gram", "kg": "kilogram",
    
    # Tiempo
    "s": "second", "min": "minutes", "h": "hour",
    
    # Datos
    "b": "bit", "kb": "kilobit", "mb": "megabit", 
    "gb": "gigabit", "tb": "terabit",
    
    # Velocidad
    "mph": "mile per hour", "kph": "kilometer per hour",
    
    # Temperatura
    "°c": "degree celsius", "°f": "degree fahrenheit",
    
    # Y muchos más...
}
```

**Configuración**:
```python
opciones_normalizacion = NormalizationOptions(
    unit_normalization=True  # Habilitar procesamiento de unidades
)
```

---

#### 6. Normalización de Tiempo

Convierte expresiones de tiempo a formato hablado.

**Estrategia**:
- Analizar formato HH:MM o HH:MM:SS
- Manejar indicadores AM/PM
- Manejo especial para en punto (o'clock)

**Ejemplos**:

```python
# Entrada:  "Reunión a las 10:35 pm"
# Salida:   "Reunión a las ten thirty-five pm"

# Entrada:  "Salida 5:03 am"
# Salida:   "Salida five oh three am"

# Entrada:  "Son las 3:00"
# Salida:   "Son las three o'clock"
```

---

#### 7. Normalización de Números Telefónicos

Convierte números telefónicos a grupos de dígitos hablados.

**Estrategia**:
- Analizar código de país, código de área, prefijo, número de línea
- Agrupar dígitos para habla natural
- Manejar código de país opcional

**Ejemplos**:

```python
# Entrada:  "Llama al (555) 123-4567"
# Salida:   "Llama al five five five, one two three, four five six seven"

# Entrada:  "+1 (800) 555-0123"
# Salida:   "+one, eight zero zero, five five five, zero one two three"
```

**Configuración**:
```python
opciones_normalizacion = NormalizationOptions(
    phone_normalization=True  # Habilitar procesamiento de teléfonos
)
```

---

### Sanitización de Caracteres

**Ubicación**: `api/src/services/text_processing/normalizer.py`

Después de la normalización, se manejan los caracteres especiales restantes.

#### Reemplazo de Símbolos

**Estrategia**: Reemplazar símbolos con equivalentes hablados o eliminarlos.

```python
SYMBOL_REPLACEMENTS = {
    '~': ' ',
    '@': ' at ',
    '#': ' number ',
    '$': ' dollar ',
    '%': ' percent ',
    '^': ' ',
    '&': ' and ',
    '*': ' ',
    '_': ' ',
    '|': ' ',
    '\\': ' ',
    '/': ' slash ',
    '=': ' equals ',
    '+': ' plus ',
}
```

**Ejemplos**:

```python
# Entrada:  "Compra productos @ tienda & online"
# Salida:   "Compra productos at tienda and online"

# Entrada:  "Descuento: 20% de descuento"
# Salida:   "Descuento: twenty percent de descuento"

# Entrada:  "Fórmula: x + y = z"
# Salida:   "Fórmula: x plus y equals z"
```

**Configuración**:
```python
opciones_normalizacion = NormalizationOptions(
    replace_remaining_symbols=True  # Habilitar reemplazo de símbolos
)
```

---

#### Normalización de Comillas y Paréntesis

**Estrategia**: Estandarizar varios estilos de comillas a comillas ASCII básicas.

```python
# Comillas Unicode a ASCII
texto = texto.replace(chr(8216), "'")  # ' → '
texto = texto.replace(chr(8217), "'")  # ' → '
texto = texto.replace(chr(8220), '"')  # " → "
texto = texto.replace(chr(8221), '"')  # " → "

# Guillemets a comillas
texto = texto.replace("«", '"')
texto = texto.replace("»", '"')
```

**Ejemplos**:

```python
# Entrada:  "Él dijo 'hola'"
# Salida:   "Él dijo 'hola'"

# Entrada:  "Título del libro: «Ejemplo»"
# Salida:   "Título del libro: "Ejemplo""
```

---

#### Puntuación CJK

**Estrategia**: Convertir puntuación china/japonesa a equivalentes occidentales.

```python
# Puntuación CJK a occidental
reemplazos = {
    '、': ', ',  # Coma de enumeración
    '。': '. ',  # Punto
    '！': '! ',  # Exclamación
    '，': ', ',  # Coma
    '：': ': ',  # Dos puntos
    '；': '; ',  # Punto y coma
    '？': '? ',  # Signo de interrogación
    '–': '- ',   # Guion medio
}
```

---

#### Normalización de Espacios en Blanco

**Estrategia**: Limpiar y estandarizar todos los espacios en blanco.

```python
# Eliminar espacios en blanco no estándar (tabulaciones, etc.)
texto = re.sub(r"[^\S \n]", " ", texto)

# Colapsar múltiples espacios
texto = re.sub(r"  +", " ", texto)

# Eliminar espacios en líneas en blanco
texto = re.sub(r"(?<=\n) +(?=\n)", "", texto)

# Convertir saltos de línea a espacios
texto = texto.replace('\n', ' ')
texto = texto.replace('\r', ' ')
```

---

#### Expansión de Abreviaturas

**Estrategia**: Expandir abreviaturas comunes para mejor pronunciación.

```python
# Títulos
texto = re.sub(r"\bDr\.(?= [A-Z])", "Doctor", texto)
texto = re.sub(r"\bMr\.(?= [A-Z])", "Mister", texto)
texto = re.sub(r"\bMs\.(?= [A-Z])", "Miss", texto)
texto = re.sub(r"\bMrs\.(?= [A-Z])", "Mrs", texto)
texto = re.sub(r"\betc\.(?! [A-Z])", "etc", texto)
```

**Ejemplos**:

```python
# Entrada:  "Dr. Smith está aquí"
# Salida:   "Doctor Smith está aquí"

# Entrada:  "Mr. Johnson, Ms. Lee, etc."
# Salida:   "Mister Johnson, Miss Lee, etc"
```

---

### Opciones de Configuración

Todas las características de sanitización pueden configurarse independientemente usando el esquema `NormalizationOptions`.

#### Esquema de Configuración Completo

```python
class NormalizationOptions(BaseModel):
    """Opciones de configuración de sanitización"""
    
    normalize: bool = Field(
        default=True,
        description="Interruptor maestro para toda normalización"
    )
    
    url_normalization: bool = Field(
        default=True,
        description="Convertir URLs a formato pronunciable"
    )
    
    email_normalization: bool = Field(
        default=True,
        description="Convertir emails a formato pronunciable"
    )
    
    unit_normalization: bool = Field(
        default=False,
        description="Convertir unidades (10KB → diez kilobytes)"
    )
    
    phone_normalization: bool = Field(
        default=True,
        description="Convertir números telefónicos a dígitos hablados"
    )
    
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Expandir (s) a s"
    )
    
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Reemplazar símbolos con palabras"
    )
```

#### Ejemplos de Uso

**Normalización Completa (Predeterminada)**:
```python
opciones = NormalizationOptions()
# Todas las características habilitadas
```

**Normalización Mínima**:
```python
opciones = NormalizationOptions(
    normalize=True,
    url_normalization=False,
    email_normalization=False,
    phone_normalization=False,
    replace_remaining_symbols=False
)
# Solo normalización básica
```

**Configuración Personalizada**:
```python
opciones = NormalizationOptions(
    normalize=True,
    url_normalization=True,
    unit_normalization=True,  # Habilitar conversión de unidades
    email_normalization=False,
    phone_normalization=False
)
# Solo URLs y unidades
```

**Sin Normalización**:
```python
opciones = NormalizationOptions(
    normalize=False
)
# Omitir toda normalización
```

---

### Ejemplos Prácticos

#### Ejemplo 1: Documentación Técnica

**Entrada**:
```
Visita https://api.ejemplo.com:8080/docs para documentación.
Contacta soporte@ejemplo.com si necesitas ayuda.
La API procesa 1.5k solicitudes/seg con 99.9% tiempo activo.
```

**Configuración**:
```python
opciones = NormalizationOptions(
    normalize=True,
    url_normalization=True,
    email_normalization=True,
    unit_normalization=True
)
```

**Salida**:
```
Visita https api dot ejemplo dot com colon eighty eighty slash docs para documentación.
Contacta soporte at ejemplo dot com si necesitas ayuda.
La API procesa one point five thousand solicitudes slash seg con ninety-nine point nine percent tiempo activo.
```

---

#### Ejemplo 2: Producto de E-commerce

**Entrada**:
```
Producto #12345: Auriculares Premium
Precio: $249.99 (¡20% de descuento!)
Peso: 250g
Envío: Gratis para pedidos $50+
Llama al (800) 555-0199 para ordenar
```

**Configuración**:
```python
opciones = NormalizationOptions(
    normalize=True,
    unit_normalization=True,
    phone_normalization=True
)
```

**Salida**:
```
Producto number twelve thousand, three hundred and forty-five: Auriculares Premium
Precio: two hundred and forty-nine dollars and ninety-nine cents (¡twenty percent de descuento!)
Peso: two hundred and fifty grams
Envío: Gratis para pedidos fifty dollars plus
Llama al eight zero zero, five five five, zero one nine nine para ordenar
```

---

### Adaptación a Tu Proyecto

#### Paso 1: Comprender Tus Requisitos

Pregúntate:
- ¿Qué tipos de entrada recibiré?
- ¿Necesito sanitización de URL/email?
- ¿Son comunes las unidades y mediciones?
- ¿Qué idioma(s) soportaré?

#### Paso 2: Extraer Componentes Principales

El sistema de sanitización tiene componentes modulares que puedes adaptar:

**Módulo de Normalización de Texto** (`normalizer.py`):
- Patrones de expresiones regulares
- Funciones manejadoras para cada tipo
- Procesamiento basado en configuración

**Capa de Validación** (`schemas.py`):
- Modelos Pydantic para seguridad de tipos
- Validadores de campos
- Esquemas de configuración

**Pipeline de Procesamiento** (`text_processor.py`):
- Lógica de fragmentación de texto
- Aplicación de normalización
- Generación de tokens

#### Paso 3: Personalizar Patrones

Modifica patrones regex para tus necesidades:

```python
# Agregar nuevo símbolo de moneda
MONEY_UNITS = {
    "$": ("dollar", "cent"),
    "£": ("pound", "pence"),
    "€": ("euro", "cent"),
    "¥": ("yen", "sen"),  # Agregar yen
    "₱": ("peso", "centavo"),  # Agregar peso
}

# Agregar nuevas unidades
VALID_UNITS = {
    # ... unidades existentes ...
    "bar": "bar",        # Unidad de presión
    "psi": "pounds per square inch",
}

# Agregar nuevos TLDs para URLs
VALID_TLDS = [
    # ... TLDs existentes ...
    "app", "dev", "cloud", "mx", "es",
]
```

#### Paso 4: Crear Manejadores Específicos de Idioma

Para soporte multilingüe:

```python
def normalize_text_with_language(text: str, lang: str, options):
    """Normalización consciente del idioma"""
    
    # Aplicar reglas específicas del idioma
    if lang in ["es", "spanish", "español"]:
        # Manejo específico de español
        text = handle_spanish_abbreviations(text)
        text = handle_spanish_numbers(text)
    
    elif lang in ["en", "english", "inglés"]:
        # Manejo específico de inglés
        text = normalize_text(text, options)
    
    return text

def handle_spanish_numbers(text: str) -> str:
    """Convertir formatos de números españoles"""
    # 1.000.000 (Europeo) → 1000000
    text = re.sub(r"(\d+)\.(\d{3})", r"\1\2", text)
    
    # 5,50 (decimal) → 5.50
    text = re.sub(r"(\d+),(\d+)", r"\1.\2", text)
    
    return text

def handle_spanish_abbreviations(text: str) -> str:
    """Expandir abreviaturas españolas"""
    # Títulos
    text = re.sub(r"\bDr\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\bDra\.(?= [A-Z])", "Doctora", text)
    text = re.sub(r"\bSr\.(?= [A-Z])", "Señor", text)
    text = re.sub(r"\bSra\.(?= [A-Z])", "Señora", text)
    text = re.sub(r"\bSrta\.(?= [A-Z])", "Señorita", text)
    
    return text
```

#### Paso 5: Implementar Mejora Progresiva

Comienza simple, agrega características según sea necesario:

```python
# Fase 1: Solo validación básica
class MinimalRequest(BaseModel):
    text: str
    # Sin normalización todavía

# Fase 2: Agregar normalización configurable
class BasicRequest(BaseModel):
    text: str
    normalize_urls: bool = True
    normalize_numbers: bool = True

# Fase 3: Completo (como Kokoro-FastAPI)
class FullRequest(BaseModel):
    input: str
    normalization_options: NormalizationOptions = NormalizationOptions()
```

#### Paso 6: Estrategia de Pruebas

Crear suites de pruebas comprensivas:

```python
def test_sanitization_suite():
    """Probar varios escenarios de sanitización"""
    
    casos_prueba = [
        # URLs
        ("Visita https://ejemplo.com", 
         "Visita https ejemplo dot com"),
        
        # Dinero
        ("Cuesta $50.30", 
         "Cuesta fifty dollars and thirty cents"),
        
        # Caracteres especiales
        ("A & B @ tienda", 
         "A and B at tienda"),
        
        # Casos límite
        ("", ""),  # Cadena vacía
        ("   ", ""),  # Solo espacios en blanco
        ("123", "one hundred and twenty-three"),
    ]
    
    for texto_entrada, esperado in casos_prueba:
        resultado = normalize_text(texto_entrada, NormalizationOptions())
        assert resultado == esperado, f"Falló: {texto_entrada}"
```

#### Paso 7: Optimización de Rendimiento

Para uso en producción:

```python
# 1. Pre-compilar patrones regex
URL_PATTERN = re.compile(r"...", re.IGNORECASE)  # Compilar una vez

# 2. Usar caché para conversiones repetidas
@lru_cache(maxsize=1000)
def number_to_words_cached(num: int) -> str:
    return inflect_engine.number_to_words(num)

# 3. Inicialización perezosa de recursos pesados
class TextNormalizer:
    _inflect_engine = None
    
    @classmethod
    def get_engine(cls):
        if cls._inflect_engine is None:
            cls._inflect_engine = inflect.engine()
        return cls._inflect_engine
```

#### Paso 8: Consideraciones de Seguridad

Implementar mejores prácticas de seguridad:

```python
# 1. Límites de longitud de entrada
class SecureRequest(BaseModel):
    input: str = Field(..., max_length=10000)  # Limitar longitud

# 2. Lista blanca de caracteres (si es necesario)
ALLOWED_CHARS = set(string.ascii_letters + string.digits + 
                    string.punctuation + " \n\t" + 
                    "áéíóúñÁÉÍÓÚÑ¿¡")  # Incluir caracteres españoles

def sanitize_input(text: str) -> str:
    """Eliminar caracteres potencialmente peligrosos"""
    return "".join(c for c in text if c in ALLOWED_CHARS)

# 3. Limitación de tasa en operaciones costosas
from functools import wraps
import time

def rate_limit(max_per_minute=60):
    """Decorador para limitar llamadas a función"""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Eliminar llamadas antiguas
            calls[:] = [t for t in calls if now - t < 60]
            
            if len(calls) >= max_per_minute:
                raise Exception("Límite de tasa excedido")
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

### Resumen de Mejores Prácticas

1. **Validación Primero**: Siempre validar entrada antes de procesar
2. **Fallar de Forma Segura**: Proporcionar respaldos para fallos de normalización
3. **Configuración Sobre Código**: Hacer características configurables
4. **Probar Extensivamente**: Cubrir casos límite y entrada malformada
5. **Documentar Comportamiento**: Explicar qué hace cada normalización
6. **El Rendimiento Importa**: Pre-compilar patrones, cachear resultados
7. **Conciencia de Seguridad**: Limitar tamaño de entrada, sanitizar caracteres peligrosos
8. **Diseño Modular**: Mantener funciones de normalización independientes
9. **Soporte de Idiomas**: Manejar texto multilingüe apropiadamente
10. **Mejora Progresiva**: Comenzar simple, agregar características incrementalmente

---

## Conclusion / Conclusión

This guide provides a comprehensive overview of text sanitization strategies that can be adapted to any text processing project. The modular design allows you to pick and choose which features you need while maintaining security and performance.

Esta guía proporciona una visión general completa de las estrategias de sanitización de texto que pueden adaptarse a cualquier proyecto de procesamiento de texto. El diseño modular te permite elegir qué características necesitas mientras mantienes seguridad y rendimiento.

---

**License / Licencia**: Apache 2.0  
**Project / Proyecto**: Kokoro-FastAPI  
**Repository / Repositorio**: https://github.com/groxaxo/Kokoro-FastAPI
