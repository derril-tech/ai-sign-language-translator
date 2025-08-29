// GDPR Compliance System with Audit Logs and Data Rights Management

interface PersonalData {
  id: string;
  type: 'transcript' | 'video' | 'audio' | 'biometric' | 'metadata';
  content: any;
  subject: string; // Data subject identifier
  purpose: string; // Purpose of processing
  legalBasis: 'consent' | 'contract' | 'legal_obligation' | 'vital_interests' | 'public_task' | 'legitimate_interests';
  retention: {
    period: number; // Days
    reason: string;
    autoDelete: boolean;
  };
  created: number;
  lastAccessed: number;
  encrypted: boolean;
  anonymized: boolean;
  shared: boolean;
  location: 'local' | 'cloud' | 'both';
}

interface ConsentRecord {
  id: string;
  subject: string;
  purpose: string;
  granted: boolean;
  timestamp: number;
  version: string;
  method: 'explicit' | 'implied' | 'opt_in' | 'opt_out';
  evidence: {
    ipAddress?: string;
    userAgent?: string;
    consentText: string;
    signature?: string;
  };
  withdrawn?: {
    timestamp: number;
    method: string;
  };
}

interface AuditLogEntry {
  id: string;
  timestamp: number;
  action: 'create' | 'read' | 'update' | 'delete' | 'export' | 'anonymize' | 'share' | 'consent' | 'withdraw';
  subject: string;
  dataType: string;
  dataId?: string;
  actor: string; // Who performed the action
  purpose: string;
  legalBasis: string;
  details: Record<string, any>;
  ipAddress?: string;
  userAgent?: string;
  result: 'success' | 'failure' | 'partial';
  error?: string;
}

interface DataSubjectRequest {
  id: string;
  subject: string;
  type: 'access' | 'rectification' | 'erasure' | 'portability' | 'restriction' | 'objection';
  status: 'pending' | 'processing' | 'completed' | 'rejected';
  submitted: number;
  deadline: number; // 30 days from submission
  completed?: number;
  details: Record<string, any>;
  response?: {
    data?: any;
    explanation: string;
    actions: string[];
  };
}

export class GDPRCompliance {
  private personalData: Map<string, PersonalData> = new Map();
  private consentRecords: Map<string, ConsentRecord> = new Map();
  private auditLog: AuditLogEntry[] = [];
  private dataSubjectRequests: Map<string, DataSubjectRequest> = new Map();
  private retentionPolicies: Map<string, { period: number; reason: string }> = new Map();

  constructor() {
    this.initializeRetentionPolicies();
    this.startRetentionCleanup();
  }

  // Initialize default retention policies
  private initializeRetentionPolicies(): void {
    this.retentionPolicies.set('transcript', { period: 365, reason: 'Service improvement and user history' });
    this.retentionPolicies.set('video', { period: 30, reason: 'Technical debugging and quality assurance' });
    this.retentionPolicies.set('audio', { period: 30, reason: 'Technical debugging and quality assurance' });
    this.retentionPolicies.set('biometric', { period: 90, reason: 'Authentication and security' });
    this.retentionPolicies.set('metadata', { period: 730, reason: 'Analytics and service improvement' });
  }

  // Record personal data processing
  public async recordPersonalData(
    type: PersonalData['type'],
    content: any,
    subject: string,
    purpose: string,
    legalBasis: PersonalData['legalBasis'],
    location: PersonalData['location'] = 'local'
  ): Promise<string> {
    const dataId = `data_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const retentionPolicy = this.retentionPolicies.get(type);

    const personalData: PersonalData = {
      id: dataId,
      type,
      content: await this.encryptIfRequired(content, type),
      subject,
      purpose,
      legalBasis,
      retention: {
        period: retentionPolicy?.period || 365,
        reason: retentionPolicy?.reason || 'Default retention',
        autoDelete: true
      },
      created: Date.now(),
      lastAccessed: Date.now(),
      encrypted: this.shouldEncrypt(type),
      anonymized: false,
      shared: false,
      location
    };

    this.personalData.set(dataId, personalData);

    // Log the data creation
    await this.logAuditEvent({
      action: 'create',
      subject,
      dataType: type,
      dataId,
      actor: 'system',
      purpose,
      legalBasis,
      details: {
        location,
        encrypted: personalData.encrypted,
        retentionPeriod: personalData.retention.period
      },
      result: 'success'
    });

    return dataId;
  }

  // Record consent
  public async recordConsent(
    subject: string,
    purpose: string,
    granted: boolean,
    method: ConsentRecord['method'],
    consentText: string,
    evidence: Partial<ConsentRecord['evidence']> = {}
  ): Promise<string> {
    const consentId = `consent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const consentRecord: ConsentRecord = {
      id: consentId,
      subject,
      purpose,
      granted,
      timestamp: Date.now(),
      version: '1.0',
      method,
      evidence: {
        consentText,
        ...evidence
      }
    };

    this.consentRecords.set(consentId, consentRecord);

    await this.logAuditEvent({
      action: 'consent',
      subject,
      dataType: 'consent',
      dataId: consentId,
      actor: subject,
      purpose,
      legalBasis: 'consent',
      details: {
        granted,
        method,
        version: consentRecord.version
      },
      result: 'success'
    });

    return consentId;
  }

  // Withdraw consent
  public async withdrawConsent(
    subject: string,
    purpose: string,
    method: string = 'user_request'
  ): Promise<void> {
    // Find and update consent records
    for (const [id, consent] of this.consentRecords) {
      if (consent.subject === subject && consent.purpose === purpose && consent.granted) {
        consent.withdrawn = {
          timestamp: Date.now(),
          method
        };

        await this.logAuditEvent({
          action: 'withdraw',
          subject,
          dataType: 'consent',
          dataId: id,
          actor: subject,
          purpose,
          legalBasis: 'consent',
          details: { method },
          result: 'success'
        });

        // Delete or anonymize related personal data
        await this.handleConsentWithdrawal(subject, purpose);
      }
    }
  }

  // Handle data subject access request (Article 15)
  public async handleAccessRequest(subject: string): Promise<DataSubjectRequest> {
    const requestId = `request_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const request: DataSubjectRequest = {
      id: requestId,
      subject,
      type: 'access',
      status: 'processing',
      submitted: Date.now(),
      deadline: Date.now() + (30 * 24 * 60 * 60 * 1000), // 30 days
      details: {}
    };

    this.dataSubjectRequests.set(requestId, request);

    // Collect all personal data for the subject
    const subjectData = Array.from(this.personalData.values())
      .filter(data => data.subject === subject);

    const subjectConsents = Array.from(this.consentRecords.values())
      .filter(consent => consent.subject === subject);

    const subjectAuditLogs = this.auditLog
      .filter(log => log.subject === subject);

    const response = {
      data: {
        personalData: subjectData.map(data => ({
          id: data.id,
          type: data.type,
          purpose: data.purpose,
          legalBasis: data.legalBasis,
          created: new Date(data.created).toISOString(),
          lastAccessed: new Date(data.lastAccessed).toISOString(),
          retention: data.retention,
          encrypted: data.encrypted,
          anonymized: data.anonymized,
          location: data.location
        })),
        consents: subjectConsents,
        auditLog: subjectAuditLogs
      },
      explanation: 'Complete record of personal data processing',
      actions: ['Data exported in machine-readable format']
    };

    request.response = response;
    request.status = 'completed';
    request.completed = Date.now();

    await this.logAuditEvent({
      action: 'export',
      subject,
      dataType: 'all',
      actor: subject,
      purpose: 'data_subject_access',
      legalBasis: 'legal_obligation',
      details: {
        requestId,
        dataItems: subjectData.length,
        consentRecords: subjectConsents.length,
        auditEntries: subjectAuditLogs.length
      },
      result: 'success'
    });

    return request;
  }

  // Handle right to erasure request (Article 17)
  public async handleErasureRequest(
    subject: string,
    reason: string = 'user_request'
  ): Promise<DataSubjectRequest> {
    const requestId = `erasure_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const request: DataSubjectRequest = {
      id: requestId,
      subject,
      type: 'erasure',
      status: 'processing',
      submitted: Date.now(),
      deadline: Date.now() + (30 * 24 * 60 * 60 * 1000),
      details: { reason }
    };

    this.dataSubjectRequests.set(requestId, request);

    // Check if erasure is possible
    const canErase = await this.checkErasureEligibility(subject);
    
    if (canErase.eligible) {
      // Perform erasure
      const deletedItems = await this.eraseSubjectData(subject);
      
      request.response = {
        explanation: 'Personal data has been erased',
        actions: [
          `Deleted ${deletedItems.personalData} personal data items`,
          `Anonymized ${deletedItems.auditLogs} audit log entries`,
          `Marked ${deletedItems.consents} consent records as withdrawn`
        ]
      };
      request.status = 'completed';
    } else {
      request.response = {
        explanation: `Erasure not possible: ${canErase.reason}`,
        actions: []
      };
      request.status = 'rejected';
    }

    request.completed = Date.now();

    await this.logAuditEvent({
      action: 'delete',
      subject,
      dataType: 'all',
      actor: subject,
      purpose: 'right_to_erasure',
      legalBasis: 'legal_obligation',
      details: {
        requestId,
        reason,
        eligible: canErase.eligible,
        result: request.status
      },
      result: canErase.eligible ? 'success' : 'failure'
    });

    return request;
  }

  // Handle data portability request (Article 20)
  public async handlePortabilityRequest(subject: string): Promise<DataSubjectRequest> {
    const requestId = `portability_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const request: DataSubjectRequest = {
      id: requestId,
      subject,
      type: 'portability',
      status: 'processing',
      submitted: Date.now(),
      deadline: Date.now() + (30 * 24 * 60 * 60 * 1000),
      details: {}
    };

    this.dataSubjectRequests.set(requestId, request);

    // Export data in structured format
    const portableData = await this.exportPortableData(subject);

    request.response = {
      data: portableData,
      explanation: 'Data exported in machine-readable format',
      actions: ['Data package created for transfer']
    };
    request.status = 'completed';
    request.completed = Date.now();

    await this.logAuditEvent({
      action: 'export',
      subject,
      dataType: 'portable',
      actor: subject,
      purpose: 'data_portability',
      legalBasis: 'legal_obligation',
      details: {
        requestId,
        format: 'JSON',
        size: JSON.stringify(portableData).length
      },
      result: 'success'
    });

    return request;
  }

  // Log audit event
  private async logAuditEvent(event: Omit<AuditLogEntry, 'id' | 'timestamp'>): Promise<void> {
    const auditEntry: AuditLogEntry = {
      id: `audit_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      ...event
    };

    this.auditLog.push(auditEntry);

    // Keep audit log size manageable (last 10000 entries)
    if (this.auditLog.length > 10000) {
      this.auditLog = this.auditLog.slice(-10000);
    }
  }

  // Check if data should be encrypted
  private shouldEncrypt(type: PersonalData['type']): boolean {
    return ['biometric', 'video', 'audio'].includes(type);
  }

  // Encrypt content if required
  private async encryptIfRequired(content: any, type: PersonalData['type']): Promise<any> {
    if (this.shouldEncrypt(type)) {
      // In production, use proper encryption
      return btoa(JSON.stringify(content)); // Simple base64 encoding for demo
    }
    return content;
  }

  // Handle consent withdrawal
  private async handleConsentWithdrawal(subject: string, purpose: string): Promise<void> {
    // Find and handle personal data that relied on this consent
    for (const [id, data] of this.personalData) {
      if (data.subject === subject && data.purpose === purpose && data.legalBasis === 'consent') {
        // Check if we have another legal basis
        const hasOtherBasis = await this.checkAlternateLegalBasis(data);
        
        if (!hasOtherBasis) {
          // Delete or anonymize the data
          if (data.type === 'transcript' || data.type === 'metadata') {
            // Anonymize instead of delete for these types
            await this.anonymizeData(id);
          } else {
            // Delete other types
            await this.deleteData(id);
          }
        }
      }
    }
  }

  // Check erasure eligibility
  private async checkErasureEligibility(subject: string): Promise<{ eligible: boolean; reason?: string }> {
    // Check for legal obligations to retain data
    const hasLegalObligation = Array.from(this.personalData.values())
      .some(data => data.subject === subject && data.legalBasis === 'legal_obligation');

    if (hasLegalObligation) {
      return { eligible: false, reason: 'Legal obligation to retain data' };
    }

    // Check for ongoing contracts
    const hasContract = Array.from(this.personalData.values())
      .some(data => data.subject === subject && data.legalBasis === 'contract');

    if (hasContract) {
      return { eligible: false, reason: 'Data needed for contract performance' };
    }

    return { eligible: true };
  }

  // Erase subject data
  private async eraseSubjectData(subject: string): Promise<{
    personalData: number;
    auditLogs: number;
    consents: number;
  }> {
    let deletedPersonalData = 0;
    let anonymizedAuditLogs = 0;
    let withdrawnConsents = 0;

    // Delete personal data
    for (const [id, data] of this.personalData) {
      if (data.subject === subject) {
        this.personalData.delete(id);
        deletedPersonalData++;
      }
    }

    // Anonymize audit logs (can't delete for compliance)
    for (const log of this.auditLog) {
      if (log.subject === subject) {
        log.subject = '[ANONYMIZED]';
        if (log.details.subject) {
          log.details.subject = '[ANONYMIZED]';
        }
        anonymizedAuditLogs++;
      }
    }

    // Mark consents as withdrawn
    for (const [id, consent] of this.consentRecords) {
      if (consent.subject === subject && !consent.withdrawn) {
        consent.withdrawn = {
          timestamp: Date.now(),
          method: 'erasure_request'
        };
        withdrawnConsents++;
      }
    }

    return {
      personalData: deletedPersonalData,
      auditLogs: anonymizedAuditLogs,
      consents: withdrawnConsents
    };
  }

  // Export portable data
  private async exportPortableData(subject: string): Promise<any> {
    const subjectData = Array.from(this.personalData.values())
      .filter(data => data.subject === subject && ['consent', 'contract'].includes(data.legalBasis));

    return {
      export_info: {
        subject,
        generated: new Date().toISOString(),
        format: 'JSON',
        version: '1.0'
      },
      personal_data: subjectData.map(data => ({
        type: data.type,
        content: data.encrypted ? '[ENCRYPTED]' : data.content,
        purpose: data.purpose,
        created: new Date(data.created).toISOString(),
        last_accessed: new Date(data.lastAccessed).toISOString()
      })),
      consents: Array.from(this.consentRecords.values())
        .filter(consent => consent.subject === subject)
        .map(consent => ({
          purpose: consent.purpose,
          granted: consent.granted,
          timestamp: new Date(consent.timestamp).toISOString(),
          method: consent.method,
          withdrawn: consent.withdrawn ? new Date(consent.withdrawn.timestamp).toISOString() : null
        }))
    };
  }

  // Check for alternate legal basis
  private async checkAlternateLegalBasis(data: PersonalData): Promise<boolean> {
    // In production, this would check business rules
    return false; // Simplified for demo
  }

  // Anonymize data
  private async anonymizeData(dataId: string): Promise<void> {
    const data = this.personalData.get(dataId);
    if (data) {
      data.anonymized = true;
      data.subject = '[ANONYMIZED]';
      data.content = await this.anonymizeContent(data.content, data.type);
      
      await this.logAuditEvent({
        action: 'anonymize',
        subject: '[ANONYMIZED]',
        dataType: data.type,
        dataId,
        actor: 'system',
        purpose: 'consent_withdrawal',
        legalBasis: data.legalBasis,
        details: {},
        result: 'success'
      });
    }
  }

  // Delete data
  private async deleteData(dataId: string): Promise<void> {
    const data = this.personalData.get(dataId);
    if (data) {
      this.personalData.delete(dataId);
      
      await this.logAuditEvent({
        action: 'delete',
        subject: data.subject,
        dataType: data.type,
        dataId,
        actor: 'system',
        purpose: 'consent_withdrawal',
        legalBasis: data.legalBasis,
        details: {},
        result: 'success'
      });
    }
  }

  // Anonymize content
  private async anonymizeContent(content: any, type: PersonalData['type']): Promise<any> {
    if (type === 'transcript') {
      // Remove identifying information from transcript
      return content.replace(/\b[A-Z][a-z]+ [A-Z][a-z]+\b/g, '[NAME]')
                   .replace(/\b\d{3}-\d{3}-\d{4}\b/g, '[PHONE]')
                   .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '[EMAIL]');
    }
    return '[ANONYMIZED]';
  }

  // Start retention cleanup process
  private startRetentionCleanup(): void {
    // Run cleanup every 24 hours
    setInterval(() => {
      this.performRetentionCleanup();
    }, 24 * 60 * 60 * 1000);
  }

  // Perform retention cleanup
  private async performRetentionCleanup(): Promise<void> {
    const now = Date.now();
    let deletedCount = 0;

    for (const [id, data] of this.personalData) {
      if (data.retention.autoDelete) {
        const retentionPeriod = data.retention.period * 24 * 60 * 60 * 1000; // Convert days to ms
        const expiryTime = data.created + retentionPeriod;

        if (now > expiryTime) {
          this.personalData.delete(id);
          deletedCount++;

          await this.logAuditEvent({
            action: 'delete',
            subject: data.subject,
            dataType: data.type,
            dataId: id,
            actor: 'system',
            purpose: 'retention_policy',
            legalBasis: data.legalBasis,
            details: {
              retentionPeriod: data.retention.period,
              reason: data.retention.reason
            },
            result: 'success'
          });
        }
      }
    }

    if (deletedCount > 0) {
      console.log(`GDPR retention cleanup: deleted ${deletedCount} expired data items`);
    }
  }

  // Get compliance status
  public getComplianceStatus(): {
    totalDataItems: number;
    encryptedItems: number;
    anonymizedItems: number;
    consentRecords: number;
    auditLogEntries: number;
    pendingRequests: number;
  } {
    const personalDataArray = Array.from(this.personalData.values());
    
    return {
      totalDataItems: personalDataArray.length,
      encryptedItems: personalDataArray.filter(d => d.encrypted).length,
      anonymizedItems: personalDataArray.filter(d => d.anonymized).length,
      consentRecords: this.consentRecords.size,
      auditLogEntries: this.auditLog.length,
      pendingRequests: Array.from(this.dataSubjectRequests.values())
        .filter(r => r.status === 'pending' || r.status === 'processing').length
    };
  }

  // Export audit log
  public exportAuditLog(startDate?: Date, endDate?: Date): AuditLogEntry[] {
    let logs = this.auditLog;

    if (startDate || endDate) {
      logs = logs.filter(log => {
        const logDate = new Date(log.timestamp);
        if (startDate && logDate < startDate) return false;
        if (endDate && logDate > endDate) return false;
        return true;
      });
    }

    return logs;
  }

  // Clear all data (for testing/development)
  public clearAllData(): void {
    this.personalData.clear();
    this.consentRecords.clear();
    this.auditLog.length = 0;
    this.dataSubjectRequests.clear();
  }
}
