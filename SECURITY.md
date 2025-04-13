# Security Policy

VitaSort prioritizes data security and compliance, especially given its handling of sensitive resume data (e.g., PII, skills, experience). Below are the key security measures implemented to protect users and ensure enterprise-grade reliability.

## Security Features

- **PII Redaction:**
  - Utilizes regex-based detection to identify and mask personally identifiable information (PII) such as emails, phone numbers, and Social Security Numbers (SSNs) before processing or storage.
  - Example: `re.compile(r'[\w\.-]+@[\w\.-]+')` for email redaction in Python.
- **GDPR Compliance Tools:**
  - Implements automated audits for data retention, ensuring compliance with General Data Protection Regulation (GDPR) requirements.
  - Provides consent management features, allowing users to control data usage and request deletion via the collaboration hub.
  - Regularly updates compliance checks based on EU standards (e.g., Article 17 right to erasure).
- **Role-Based Access Control (RBAC):**
  - Employs JSON Web Tokens (JWT) for authentication, restricting access to authorized users (e.g., HR admins, job seekers).
  - Defines roles (e.g., viewer, editor, admin) with granular permissions managed via FastAPI middleware.
- **Data Encryption:**
  - Uses AES-256 encryption for data at rest, integrated with cloud storage (AWS/GCP/Azure via CloudStorageAdapter).
  - Implements TLS 1.3 for data in transit, ensuring secure API communications.
- **Input Validation:**
  - Validates all resume inputs (PDF, DOCX, images) using schema enforcement with FastAPI’s Pydantic models to prevent injection attacks (e.g., SQL, XSS).
  - Example: 
    ```python
    class ResumeInput(BaseModel): 
        file: UploadFile = None
    ```
- **Rate Limiting:**
  - Applies rate limiting on API endpoints (e.g., `/api/v1/resume/parse`) using FastAPI’s SlowAPI to mitigate brute-force attacks, set at 100 requests/hour per IP.
- **Audit Logging:**
  - Logs all user actions (e.g., file uploads, edits) with timestamps and user IDs, stored securely in a cloud data lake for 90 days, compliant with GDPR audit trails.
  - Accessible only to admin roles via encrypted logs.

## Security Best Practices

- **Dependency Scanning:** Regularly scans dependencies (e.g., spaCy, Plotly) using tools like `pip-audit` to identify vulnerabilities.
- **Code Reviews:** Conducts peer reviews for all security-sensitive code (e.g., PII handling) using GitHub pull requests.
- **Penetration Testing:** Plans quarterly penetration tests on the API and UI, targeting common vulnerabilities (e.g., OWASP Top 10).
- **Secure Development Lifecycle (SDLC):** Integrates security checks (e.g., static analysis with `bandit`) into the CI/CD pipeline hosted on GitHub Actions.

## Compliance and Certifications

- **SOC 2 Compliance:** Aligns with SOC 2 Type II standards for data security, availability, and processing integrity, with ongoing audits planned.
- **ISO 27001 Readiness:** Follows ISO 27001 guidelines for information security management, with documentation available upon request.
- **Developer Expertise:** Built by professional, ensuring adherence to industry best practices.

## Known Limitations

- **Third-Party Risks:** Relies on Tesseract OCR and cloud providers (AWS/GCP/Azure); security depends on their compliance (e.g., AWS’s Shared Responsibility Model).
- **Emerging Threats:** Real-time market data APIs may introduce new vulnerabilities; monitoring is ongoing with automated alerts.
- **Mitigation:** Regularly updates dependencies and conducts risk assessments every 6 months.

## Reporting Security Issues

- To report vulnerabilities or security concerns, please create an issue on the GitHub repository: [https://github.com/la-b-ib/vitasort/issues](https://github.com/la-b-ib/vitasort/issues).
- For sensitive disclosures, email `labib-x@protonmail.com ` with the subject **“VitaSort Security Disclosure”**
- Response time: Within 48 hours, with a detailed resolution plan.

## Future Enhancements

- **Multi-Factor Authentication (MFA):** Planned integration for enhanced user login security.
- **AI-Driven Threat Detection:** Implementing anomaly detection to identify unusual access patterns.
