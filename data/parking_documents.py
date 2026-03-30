"""
Static parking information documents for Pinecone ingestion.

Each document has a unique 'doc_id' in metadata so evaluation
can track which docs are retrieved for a given query.
"""

from langchain_core.documents import Document

PARKING_DOCUMENTS = [
    Document(
        page_content=(
            "Slytherin Parking Facility is a modern, multi-level parking structure "
            "located in the heart of downtown. We offer 500 parking spaces across "
            "5 floors with 24/7 security surveillance, CCTV coverage on all levels, "
            "and on-site security personnel. Slytherin is fully automated with license "
            "plate recognition at entry and exit points. The facility is wheelchair "
            "accessible with elevators on every floor."
        ),
        metadata={"doc_id": "general_001", "category": "general"},
    ),
    Document(
        page_content=(
            "Slytherin is located at 42 Central Avenue, Downtown District. "
            "The nearest subway stations are Central Station (Line 1, 3-minute walk) "
            "and Commerce Square (Line 2, 5-minute walk). By bus, routes 14, 22, and "
            "57 stop directly in front of the facility. By car, take Exit 7A from "
            "the highway and follow signs for 'Slytherin Downtown'. GPS coordinates: "
            "40.7128° N, 74.0060° W."
        ),
        metadata={"doc_id": "location_001", "category": "location"},
    ),
    Document(
        page_content=(
            "Slytherin operating hours: Monday to Friday 6:00 AM – 11:00 PM, "
            "Saturday 7:00 AM – 11:00 PM, Sunday and public holidays 8:00 AM – 10:00 PM. "
            "The facility is closed on Christmas Day (December 25). "
            "24-hour access is available for monthly permit holders only. "
            "Last entry is 30 minutes before closing time."
        ),
        metadata={"doc_id": "hours_001", "category": "hours"},
    ),
    Document(
        page_content=(
            "Slytherin pricing: Standard spaces – $3/hour, $18/day maximum. "
            "Compact spaces – $2.50/hour, $15/day maximum. "
            "Electric Vehicle (EV) spaces – $3.50/hour (includes charging), $20/day maximum. "
            "Disabled/accessible spaces – $2/hour, $12/day maximum. "
            "Monthly permits: Standard $150/month, EV $200/month. "
            "Weekend flat rate: $10 all day Saturday or Sunday. "
            "Pre-booked reservations receive a 10% discount on the daily rate."
        ),
        metadata={"doc_id": "pricing_001", "category": "pricing"},
    ),
    Document(
        page_content=(
            "Slytherin offers four types of parking spaces. "
            "Standard spaces (350 total) on floors 1–4 suit regular passenger vehicles up to 5.0m long. "
            "Compact spaces (80 total) on floors 2–3 suit small cars up to 4.0m long, at a lower rate. "
            "Electric Vehicle spaces (50 total) on floor 1 include Tesla and universal Type 2 chargers; "
            "charging is included in the hourly rate. "
            "Accessible spaces (20 total) on floor 1, near the elevator, comply with ADA standards."
        ),
        metadata={"doc_id": "spaces_001", "category": "spaces"},
    ),
    Document(
        page_content=(
            "Reservation policy: Reservations can be made up to 30 days in advance and require "
            "a minimum booking period of 1 hour. To complete a reservation you need to provide: "
            "your first name, last name, vehicle license plate number, and the desired reservation period "
            "(start date/time and end date/time). Reservations are subject to administrator approval. "
            "Cancellations made 24 hours or more before the reservation start time are fully refunded. "
            "Late cancellations (under 24 hours) incur a 50% charge. No-shows are charged in full. "
            "You will receive a confirmation email once the administrator approves your reservation."
        ),
        metadata={"doc_id": "booking_001", "category": "booking"},
    ),
    Document(
        page_content=(
            "Slytherin accepted payment methods: Visa, Mastercard, American Express, contactless payments "
            "(Apple Pay, Google Pay), and cash at staffed booths on floors 1 and 2. "
            "Monthly permit holders can set up automatic bank transfers. "
            "All transactions are processed securely. Receipts are provided via email or printed on request. "
            "Lost ticket fee: $25. Replacement permit card fee: $10."
        ),
        metadata={"doc_id": "payment_001", "category": "payment"},
    ),
    Document(
        page_content=(
            "Slytherin amenities and facilities: Free Wi-Fi throughout the facility. "
            "Car wash service available on floor 1 (Monday–Saturday, 8:00 AM – 6:00 PM), prices start at $15. "
            "Tyre inflation stations on floors 1 and 3 (free of charge). "
            "CCTV and 24/7 security patrols. Emergency call points on every floor. "
            "Lost and found service at the main booth (floor 1). "
            "Bicycle storage room on floor 1 (free, limited availability, first-come first-served)."
        ),
        metadata={"doc_id": "amenities_001", "category": "amenities"},
    ),
    Document(
        page_content=(
            "Frequently Asked Questions: "
            "Q: Can I extend my reservation? Yes, extensions can be requested via the chatbot or at the booth, subject to availability. "
            "Q: What if I arrive late? Your space is held for 30 minutes after your reserved start time. "
            "Q: Is overnight parking allowed? Yes, for standard and EV monthly permit holders only. "
            "Q: Do you offer family or group discounts? No group discounts currently, but monthly permits offer the best value. "
            "Q: What happens if I lose my ticket? Report to the main booth; a $25 lost ticket fee applies. "
            "Q: Is there a height restriction? Maximum vehicle height is 2.1 metres."
        ),
        metadata={"doc_id": "faq_001", "category": "faq"},
    ),
    Document(
        page_content=(
            "Public contact information for Slytherin: "
            "Customer service phone: +1 (555) 100-2000 (available during operating hours). "
            "Customer email: info@Slytherin.example.com. "
            "Website: www.Slytherin.example.com. "
            "Address: 42 Central Avenue, Downtown District, City, 10001. "
            "For lost property: lostandfound@Slytherin.example.com."
        ),
        metadata={"doc_id": "contact_001", "category": "contact"},
    ),
]
