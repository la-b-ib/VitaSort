<xaiArtifact artifact_id="53226c9b-a8da-48ba-a10a-ad3579da0d85" artifact_version_id="b20b9ab7-7483-409a-8c8f-8edd37cedfc9" title="SECURITY.md" contentType="text/markdown">

# Security Policy for VitaSort

## Supported Versions

The following versions of VitaSort are currently supported with security updates:

| Version | Supported          |
|---------|--------------------|
| 2.3     | ✅                 |
| 2.2     | ✅                 |
| 2.1     | ❌                 |
| < 2.1   | ❌                 |

## Reporting a Vulnerability

We take the security of VitaSort seriously. If you discover a security vulnerability, please report it to us responsibly. We appreciate your efforts to help keep our project secure.

### How to Report

1. **Detailed Report**:
   - A clear description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact of the vulnerability
   - Any suggested fixes or mitigations (optional)

2. **Response Time**: 
   - We aim to acknowledge your report within 48 hours.
   - A detailed response, including next steps, will be provided within 7 business days.

3. **Confidentiality**: 
   - Please do not publicly disclose the vulnerability until we have had a chance to address it.
   - We will work with you to ensure the issue is resolved and will credit you for the discovery (unless you prefer to remain anonymous).

## Security Guidelines

To ensure the security of VitaSort, we recommend the following best practices for users and contributors:

### For Users
- **Keep Software Updated**: Always use the latest supported version of VitaSort to benefit from security patches.
- **Secure Environment**: Run VitaSort in a secure environment, such as a virtual environment, to isolate dependencies.
- **Input Validation**: Ensure all uploaded files (e.g., PDF resumes) are from trusted sources to prevent malicious file uploads.
- **Network Security**: Run VitaSort on a secure network and consider using HTTPS for any remote access to the Streamlit application.

### For Contributors
- **Dependency Management**: Regularly update and audit dependencies (e.g., `streamlit`, `PyPDF2`, `scikit-learn`) to mitigate known vulnerabilities.
- **Code Review**: All code contributions must undergo a security review to identify potential vulnerabilities, such as injection attacks or improper data handling.
- **Secure Coding Practices**:
  - Sanitize all inputs to prevent injection attacks.
  - Avoid storing sensitive data (e.g., resumes) in plain text; consider encryption for sensitive information.
  - Use secure random number generation for any cryptographic operations.

### Data Handling
- **PDF Processing**: VitaSort processes PDF resumes using `PyPDF2`. Ensure that uploaded PDFs are scanned for malicious content before processing.
- **Data Privacy**: VitaSort does not store user data by default. If implementing persistent storage, ensure compliance with data protection regulations (e.g., GDPR, CCPA).
- **Temporary Files**: Ensure temporary files created during PDF processing are securely deleted after use.

## Known Security Considerations
- **Streamlit Limitations**: Streamlit applications run on a local server by default (`http://localhost:8501`). Avoid exposing the application to public networks without proper security measures (e.g., authentication, HTTPS).
- **Third-Party Dependencies**: VitaSort relies on libraries like `PyPDF2`, `pandas`, and `scikit-learn`. Regularly check for security advisories for these dependencies.
- **File Upload Risks**: Malicious PDFs could potentially exploit vulnerabilities in `PyPDF2`. Limit file sizes to 50MB and validate file types before processing.

## Security Updates
- Security patches will be applied to supported versions (2.2 and 2.3) as needed.
- Critical vulnerabilities will be addressed with high priority, and updates will be communicated via the project's GitHub repository and release notes.

## Responsible Disclosure
We encourage security researchers to follow responsible disclosure practices. If you report a vulnerability, we will:
- Acknowledge your report promptly.
- Work with you to validate and address the issue.
- Provide credit for your contribution in our release notes or changelog (if desired).


</xaiArtifact>
