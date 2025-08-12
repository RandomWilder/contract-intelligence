"""
Contract Intelligence Layer - PoC Implementation
Focuses on party extraction and contract type classification for Hebrew contracts
"""
import openai
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


def format_rtl_safe_numbers(text: str) -> str:
    """Ensure numbers maintain LTR direction in RTL text by wrapping with Unicode directional marks"""
    # Pattern to match numbers (including dates, IDs, phone numbers, etc.)
    number_pattern = r'\b\d+[-/.\d]*\d*\b'
    
    def wrap_number(match):
        number = match.group(0)
        # Wrap with LTR override (\u202D) and pop directional formatting (\u202C)
        return f"\u202D{number}\u202C"
    
    return re.sub(number_pattern, wrap_number, text)


@dataclass
class ContractParty:
    """Represents a party in a contract"""
    name: str
    type: str  # 'company', 'individual', 'organization'
    role: str  # 'lessor', 'lessee', 'contractor', etc.
    details: Dict[str, Any]  # Additional details like ID, address, etc.
    confidence: float


@dataclass
class ContractAnalysis:
    """Results of contract intelligence analysis"""
    contract_type: str
    contract_type_confidence: float
    parties: List[ContractParty]
    key_dates: List[str]
    language: str
    analysis_timestamp: str
    processing_notes: List[str]


class ContractIntelligenceEngine:
    """
    Contract Intelligence Engine for extracting key information from contracts.
    Designed to work with Hebrew contracts while supporting other languages.
    """
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.supported_contract_types = [
            "rental_agreement",      # הסכם שכירות
            "lease_agreement",       # חוזה חכירה
            "service_agreement",     # הסכם שירותים
            "employment_contract",   # חוזה עבודה
            "partnership_agreement", # הסכם שותפות
            "sales_agreement",       # חוזה מכירה
            "consulting_agreement",  # הסכם ייעוץ
            "maintenance_agreement", # הסכם תחזוקה
            "other"
        ]
    
    def analyze_contract(self, text: str, document_name: str = "") -> ContractAnalysis:
        """
        Analyze a contract and extract key intelligence information.
        
        Args:
            text: The contract text to analyze
            document_name: Name of the source document
            
        Returns:
            ContractAnalysis object with extracted information
        """
        processing_notes = []
        
        try:
            print(f"📋 Starting contract analysis for {document_name}")
            
            # Detect language
            language = self._detect_language(text)
            processing_notes.append(f"Detected language: {language}")
            print(f"  📝 Language detected: {language}")
            
            # Extract parties
            print(f"  👥 Extracting parties...")
            parties = self._extract_parties(text, language)
            processing_notes.append(f"Extracted {len(parties)} parties")
            print(f"  ✅ Found {len(parties)} parties")
            
            # Classify contract type
            print(f"  📋 Classifying contract type...")
            contract_type, type_confidence = self._classify_contract_type(text, language)
            processing_notes.append(f"Classified as: {contract_type} (confidence: {type_confidence:.2f})")
            print(f"  ✅ Classified as: {contract_type}")
            
            # Extract key dates
            print(f"  📅 Extracting key dates...")
            key_dates = self._extract_key_dates(text, language)
            processing_notes.append(f"Found {len(key_dates)} key dates")
            print(f"  ✅ Found {len(key_dates)} key dates")
            
        except Exception as e:
            print(f"  ❌ Error during contract analysis: {e}")
            # Return basic analysis on error
            return ContractAnalysis(
                contract_type="other",
                contract_type_confidence=0.3,
                parties=[],
                key_dates=[],
                language="unknown",
                analysis_timestamp=datetime.now().isoformat(),
                processing_notes=[f"Error during analysis: {str(e)}"]
            )
        
        print(f"  🎉 Contract analysis completed successfully!")
        
        return ContractAnalysis(
            contract_type=contract_type,
            contract_type_confidence=type_confidence,
            parties=parties,
            key_dates=key_dates,
            language=language,
            analysis_timestamp=datetime.now().isoformat(),
            processing_notes=processing_notes
        )
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        # Count Hebrew characters
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0590-\u05FF]', text))
        
        if total_chars == 0:
            return "unknown"
        
        hebrew_ratio = hebrew_chars / total_chars
        
        if hebrew_ratio > 0.3:
            return "hebrew"
        elif hebrew_ratio > 0.1:
            return "mixed"
        else:
            return "english"
    
    def _extract_parties(self, text: str, language: str) -> List[ContractParty]:
        """Extract parties from contract using LLM"""
        
        # Create language-specific prompt
        if language in ["hebrew", "mixed"]:
            today_str = f"\u202D{datetime.now().strftime('%Y-%m-%d')}\u202C"
            system_prompt = f"""אתה מומחה לניתוח חוזים בעברית. המשימה שלך היא לחלץ את כל הצדדים (הגורמים) המעורבים בחוזה.

התאריך של היום: {today_str}

עבור כל צד, זהה:
1. שם הצד (חברה, אדם פרטי, או ארגון)
2. סוג הצד (company/individual/organization)
3. תפקיד בחוזה (משכיר/שוכר/קבלן/לקוח וכו')
4. פרטים נוספים (ח.פ, ת.ז, כתובת, טלפון)

חשוב: כל המספרים (תאריכים, מספרי זהות, טלפונים) חייבים להישמר בכיוון מקורי (LTR) גם בטקסט עברי.

החזר תשובה ב-JSON עם רמת ביטחון עבור כל צד."""
            
            user_prompt = f"""נתח את הטקסט הבא וחלץ את כל הצדדים בחוזה:

{text[:2000]}...

החזר JSON במבנה הבא:
{{
    "parties": [
        {{
            "name": "שם הצד",
            "type": "company/individual/organization", 
            "role": "תפקיד בחוזה",
            "details": {{"additional": "info"}},
            "confidence": 0.95
        }}
    ]
}}"""
        else:
            today_str = f"\u202D{datetime.now().strftime('%Y-%m-%d')}\u202C"
            system_prompt = f"""You are an expert contract analyst. Extract all parties involved in this contract.

Today's date is: {today_str}

For each party, identify:
1. Party name (company, individual, or organization)
2. Party type (company/individual/organization)
3. Role in contract (lessor/lessee/contractor/client etc.)
4. Additional details (registration numbers, IDs, addresses, phone numbers)

Return response in JSON format with confidence scores."""
            
            user_prompt = f"""Analyze the following contract text and extract all parties:

{text[:2000]}...

Return JSON in this structure:
{{
    "parties": [
        {{
            "name": "Party Name",
            "type": "company/individual/organization",
            "role": "role in contract",
            "details": {{"additional": "info"}},
            "confidence": 0.95
        }}
    ]
}}"""
        
        try:
            print(f"  Requesting party extraction from OpenAI...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                timeout=30  # 30 second timeout
            )
            print(f"  ✅ Received party extraction response")
            
            # Parse JSON response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response (handle markdown code blocks and extra text)
            json_str = None
            if "```json" in response_text:
                # Extract everything between ```json and ```
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
            else:
                # Fallback to any JSON-like structure
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
            
            if json_str:
                try:
                    # Clean up common JSON issues with Hebrew text
                    json_str = json_str.replace('"ל"', '"ל"')  # Fix Hebrew quote issues
                    json_str = json_str.replace('דוא"ל', 'דואל')  # Fix email field
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    print(f"Extracted JSON string: {json_str[:200]}...")
                    # Try fallback extraction
                    return self._fallback_party_extraction(text)
                
                parties = []
                for party_data in result.get("parties", []):
                    party = ContractParty(
                        name=party_data.get("name", "Unknown"),
                        type=party_data.get("type", "unknown"),
                        role=party_data.get("role", "unknown"),
                        details=party_data.get("details", {}),
                        confidence=party_data.get("confidence", 0.5)
                    )
                    parties.append(party)
                
                return parties
            
        except Exception as e:
            print(f"Error extracting parties: {e}")
            # Fallback: try to extract basic party information using regex
            return self._fallback_party_extraction(text)
        
        return []
    
    def _fallback_party_extraction(self, text: str) -> List[ContractParty]:
        """Fallback party extraction using basic patterns"""
        parties = []
        
        # Hebrew company patterns
        hebrew_company_patterns = [
            r'([א-ת\s]+בע"מ)',  # Company Ltd
            r'([א-ת\s]+ע"ר)',   # Non-profit
            r'([א-ת\s]+חברת[א-ת\s]*)',  # Company
        ]
        
        for pattern in hebrew_company_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:3]:  # Limit to 3 matches
                parties.append(ContractParty(
                    name=match.strip(),
                    type="company",
                    role="unknown",
                    details={},
                    confidence=0.6
                ))
        
        return parties
    
    def _classify_contract_type(self, text: str, language: str) -> tuple[str, float]:
        """Classify the contract type using LLM"""
        
        if language in ["hebrew", "mixed"]:
            today_str = f"\u202D{datetime.now().strftime('%Y-%m-%d')}\u202C"
            system_prompt = f"""אתה מומחה לסיווג חוזים בעברית. סווג את סוג החוזה מהרשימה הבאה:

התאריך של היום: {today_str}

- rental_agreement (הסכם שכירות)
- lease_agreement (חוזה חכירה)
- service_agreement (הסכם שירותים)
- employment_contract (חוזה עבודה)
- partnership_agreement (הסכם שותפות)
- sales_agreement (חוזה מכירה)
- consulting_agreement (הסכם ייעוץ)
- maintenance_agreement (הסכם תחזוקה)
- other (אחר)

חשוב: כל המספרים והתאריכים חייבים להישמר בכיוון מקורי (LTR).

החזר JSON עם הסיווג ורמת ביטחון."""
            
            user_prompt = f"""סווג את סוג החוזה הבא:

{text[:1500]}...

החזר JSON במבנה:
{{
    "contract_type": "rental_agreement",
    "confidence": 0.95,
    "reasoning": "הסבר קצר"
}}"""
        else:
            today_str = datetime.now().strftime('%Y-%m-%d')
            system_prompt = f"""You are an expert contract classifier. Classify the contract type from this list:

Today's date is: {today_str}

- rental_agreement
- lease_agreement  
- service_agreement
- employment_contract
- partnership_agreement
- sales_agreement
- consulting_agreement
- maintenance_agreement
- other

Return JSON with classification and confidence score."""
            
            user_prompt = f"""Classify this contract type:

{text[:1500]}...

Return JSON structure:
{{
    "contract_type": "rental_agreement",
    "confidence": 0.95,
    "reasoning": "brief explanation"
}}"""
        
        try:
            print(f"  Requesting contract classification from OpenAI...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=30  # 30 second timeout
            )
            print(f"  ✅ Received classification response")
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response (handle markdown code blocks and extra text)
            json_str = None
            if "```json" in response_text:
                # Extract everything between ```json and ```
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
            else:
                # Fallback to any JSON-like structure
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
            
            if json_str:
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error in classification: {e}")
                    print(f"Extracted JSON string: {json_str}")
                    return "other", 0.3
                
                contract_type = result.get("contract_type", "other")
                confidence = result.get("confidence", 0.5)
                
                # Validate contract type
                if contract_type not in self.supported_contract_types:
                    contract_type = "other"
                
                return contract_type, confidence
            
        except Exception as e:
            print(f"Error classifying contract: {e}")
            return "other", 0.3
        
        return "other", 0.3
    
    def _extract_key_dates(self, text: str, language: str) -> List[str]:
        """Extract key dates from the contract"""
        dates = []
        
        # Hebrew date patterns
        hebrew_date_patterns = [
            r'\d{1,2}\s+ל?חודש\s+\w+\s+\d{4}',  # 20 לחודש נובמבר 2019
            r'\d{1,2}[./]\d{1,2}[./]\d{4}',       # 20/11/2019 or 20.11.2019
            r'\d{4}-\d{1,2}-\d{1,2}',             # 2019-11-20
        ]
        
        # English date patterns
        english_date_patterns = [
            r'\d{1,2}[./]\d{1,2}[./]\d{4}',       # 11/20/2019
            r'\d{4}-\d{1,2}-\d{1,2}',             # 2019-11-20
            r'\w+\s+\d{1,2},?\s+\d{4}',           # November 20, 2019
        ]
        
        all_patterns = hebrew_date_patterns + english_date_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        # Remove duplicates and return first 10 dates
        unique_dates = list(set(dates))[:10]
        return unique_dates
    
    def create_enhanced_metadata(self, analysis: ContractAnalysis, base_metadata: Dict) -> Dict:
        """Create enhanced metadata including contract intelligence"""
        enhanced = base_metadata.copy()
        
        # Add contract intelligence fields (ChromaDB only supports str, int, float, bool)
        enhanced.update({
            "contract_type": analysis.contract_type,
            "contract_type_confidence": float(analysis.contract_type_confidence),
            "language_detected": analysis.language,
            "parties_count": len(analysis.parties),
            # Convert parties to simple strings for ChromaDB compatibility
            "parties_names": ", ".join([party.name for party in analysis.parties]) if analysis.parties else "",
            "parties_types": ", ".join([party.type for party in analysis.parties]) if analysis.parties else "",
            "parties_roles": ", ".join([party.role for party in analysis.parties]) if analysis.parties else "",
            # Convert dates to simple string
            "key_dates": ", ".join(analysis.key_dates) if analysis.key_dates else "",
            "analysis_timestamp": analysis.analysis_timestamp,
            "intelligence_version": "poc_v1.0"
        })
        
        return enhanced


def test_contract_intelligence():
    """Test function for the contract intelligence engine"""
    import os
    
    # Initialize with OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    engine = ContractIntelligenceEngine(client)
    
    # Test with sample Hebrew contract text
    sample_text = """
    הסכם שכירות בלתי מוגנת
    שנערך ונחתם בתל אביב ביום 20 לחודש נובמבר 2019
    
    בין
    חברת רחוקות בע"מ ח.פ 516051828
    כתובת רבנו תם 4 תל אביב
    (להלן: "המשכיר")
    
    לבין
    עמותת עלם ע.ר 580036945
    מרח' הירקון 35, בני ברק
    טלפון 0549773615
    (להלן: "השוכר")
    """
    
    analysis = engine.analyze_contract(sample_text, "test_contract.pdf")
    
    print("=== Contract Intelligence Analysis ===")
    print(f"Contract Type: {analysis.contract_type} (confidence: {analysis.contract_type_confidence:.2f})")
    print(f"Language: {analysis.language}")
    print(f"Parties Found: {len(analysis.parties)}")
    
    for i, party in enumerate(analysis.parties, 1):
        print(f"  Party {i}: {party.name} ({party.type}, {party.role}) - confidence: {party.confidence:.2f}")
    
    print(f"Key Dates: {analysis.key_dates}")
    print(f"Processing Notes: {analysis.processing_notes}")


if __name__ == "__main__":
    test_contract_intelligence()
