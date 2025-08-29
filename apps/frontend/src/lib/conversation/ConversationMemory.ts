// Conversation Memory System for Co-reference Continuity and Context Tracking

interface Entity {
  id: string;
  name: string;
  type: 'person' | 'place' | 'thing' | 'concept' | 'event';
  aliases: string[];
  spatialLocation?: [number, number, number]; // 3D position in signing space
  firstMention: number; // timestamp
  lastMention: number;
  mentionCount: number;
  confidence: number;
  attributes: Map<string, any>;
  relationships: Map<string, string>; // entity_id -> relationship_type
}

interface Reference {
  id: string;
  entityId: string;
  text: string;
  type: 'pronoun' | 'definite' | 'demonstrative' | 'spatial' | 'temporal';
  timestamp: number;
  spatialInfo?: {
    handedness: 'left' | 'right' | 'both';
    location: [number, number, number];
    movement?: string;
  };
  confidence: number;
}

interface ConversationTurn {
  id: string;
  speakerId: string;
  startTime: number;
  endTime: number;
  content: string;
  entities: string[]; // entity IDs mentioned in this turn
  references: string[]; // reference IDs used in this turn
  spatialSetup: Map<string, [number, number, number]>; // entity spatial assignments
}

interface TopicSegment {
  id: string;
  startTime: number;
  endTime?: number;
  mainTopic: string;
  subTopics: string[];
  entities: string[];
  coherenceScore: number;
  transitionType?: 'continuation' | 'shift' | 'return' | 'digression';
}

export class ConversationMemory {
  private entities: Map<string, Entity> = new Map();
  private references: Map<string, Reference> = new Map();
  private turns: ConversationTurn[] = [];
  private topics: TopicSegment[] = [];
  private spatialMemory: Map<string, [number, number, number]> = new Map();
  private temporalContext: Map<string, number> = new Map();
  private conversationId: string;
  private maxMemorySize: number;

  constructor(conversationId: string, maxMemorySize: number = 1000) {
    this.conversationId = conversationId;
    this.maxMemorySize = maxMemorySize;
  }

  // Add a new conversation turn
  public addTurn(
    speakerId: string,
    content: string,
    timestamp: number,
    spatialInfo?: Map<string, [number, number, number]>
  ): string {
    const turnId = `turn_${timestamp}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Extract entities and references from content
    const extractedEntities = this.extractEntities(content, timestamp);
    const extractedReferences = this.extractReferences(content, timestamp, spatialInfo);
    
    // Create turn
    const turn: ConversationTurn = {
      id: turnId,
      speakerId,
      startTime: timestamp,
      endTime: timestamp,
      content,
      entities: extractedEntities.map(e => e.id),
      references: extractedReferences.map(r => r.id),
      spatialSetup: spatialInfo || new Map()
    };
    
    this.turns.push(turn);
    
    // Update entities and references
    extractedEntities.forEach(entity => this.addOrUpdateEntity(entity));
    extractedReferences.forEach(ref => this.addReference(ref));
    
    // Update spatial memory
    if (spatialInfo) {
      spatialInfo.forEach((location, entityId) => {
        this.spatialMemory.set(entityId, location);
      });
    }
    
    // Update topic tracking
    this.updateTopicSegmentation(turn);
    
    // Maintain memory size
    this.pruneMemory();
    
    return turnId;
  }

  // Resolve a reference to its entity
  public resolveReference(
    referenceText: string,
    timestamp: number,
    spatialInfo?: { location: [number, number, number]; handedness: 'left' | 'right' | 'both' }
  ): Entity | null {
    // Try different resolution strategies
    
    // 1. Spatial resolution (for sign language)
    if (spatialInfo) {
      const spatialEntity = this.resolveSpatialReference(spatialInfo.location);
      if (spatialEntity) return spatialEntity;
    }
    
    // 2. Recency-based resolution
    const recentEntity = this.resolveByRecency(referenceText, timestamp);
    if (recentEntity) return recentEntity;
    
    // 3. Salience-based resolution
    const salientEntity = this.resolveBySalience(referenceText);
    if (salientEntity) return salientEntity;
    
    // 4. Syntactic resolution
    const syntacticEntity = this.resolveSyntactically(referenceText, timestamp);
    if (syntacticEntity) return syntacticEntity;
    
    return null;
  }

  // Get current conversation context
  public getContext(windowSize: number = 5): {
    recentTurns: ConversationTurn[];
    activeEntities: Entity[];
    currentTopic: TopicSegment | null;
    spatialSetup: Map<string, [number, number, number]>;
  } {
    const recentTurns = this.turns.slice(-windowSize);
    const activeEntityIds = new Set<string>();
    
    // Collect entities from recent turns
    recentTurns.forEach(turn => {
      turn.entities.forEach(entityId => activeEntityIds.add(entityId));
    });
    
    const activeEntities = Array.from(activeEntityIds)
      .map(id => this.entities.get(id))
      .filter((entity): entity is Entity => entity !== undefined)
      .sort((a, b) => b.lastMention - a.lastMention);
    
    const currentTopic = this.topics.length > 0 ? this.topics[this.topics.length - 1] : null;
    
    return {
      recentTurns,
      activeEntities,
      currentTopic,
      spatialSetup: new Map(this.spatialMemory)
    };
  }

  // Get entity by ID
  public getEntity(entityId: string): Entity | null {
    return this.entities.get(entityId) || null;
  }

  // Get all entities matching criteria
  public findEntities(criteria: {
    type?: string;
    name?: string;
    minConfidence?: number;
    timeRange?: [number, number];
  }): Entity[] {
    return Array.from(this.entities.values()).filter(entity => {
      if (criteria.type && entity.type !== criteria.type) return false;
      if (criteria.name && !entity.name.toLowerCase().includes(criteria.name.toLowerCase())) return false;
      if (criteria.minConfidence && entity.confidence < criteria.minConfidence) return false;
      if (criteria.timeRange) {
        const [start, end] = criteria.timeRange;
        if (entity.lastMention < start || entity.firstMention > end) return false;
      }
      return true;
    });
  }

  // Extract entities from text
  private extractEntities(content: string, timestamp: number): Entity[] {
    const entities: Entity[] = [];
    
    // Simple named entity recognition (in production, use NLP library)
    const words = content.split(/\s+/);
    const entityPatterns = [
      // Person names (capitalized words)
      /^[A-Z][a-z]+$/,
      // Places (with location indicators)
      /\b(at|in|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/g,
      // Organizations
      /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Company|Corp|Inc|LLC|University|Hospital|School)/g
    ];
    
    // Extract person names
    words.forEach((word, index) => {
      if (entityPatterns[0].test(word) && index < words.length - 1) {
        const nextWord = words[index + 1];
        if (entityPatterns[0].test(nextWord)) {
          // Likely a full name
          const fullName = `${word} ${nextWord}`;
          entities.push(this.createEntity(fullName, 'person', timestamp));
        } else if (word.length > 2) {
          // Single name
          entities.push(this.createEntity(word, 'person', timestamp));
        }
      }
    });
    
    // Extract places and organizations using regex
    entityPatterns.slice(1).forEach(pattern => {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const entityName = match[2] || match[1];
        const entityType = match[0].includes('Company') || match[0].includes('Corp') ? 'thing' : 'place';
        entities.push(this.createEntity(entityName, entityType, timestamp));
      }
    });
    
    return entities;
  }

  // Extract references from text
  private extractReferences(
    content: string,
    timestamp: number,
    spatialInfo?: Map<string, [number, number, number]>
  ): Reference[] {
    const references: Reference[] = [];
    
    // Pronoun patterns
    const pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their'];
    const demonstratives = ['this', 'that', 'these', 'those', 'here', 'there'];
    
    const words = content.toLowerCase().split(/\s+/);
    
    words.forEach((word, index) => {
      if (pronouns.includes(word)) {
        references.push(this.createReference(
          word,
          'pronoun',
          timestamp,
          spatialInfo
        ));
      } else if (demonstratives.includes(word)) {
        references.push(this.createReference(
          word,
          'demonstrative',
          timestamp,
          spatialInfo
        ));
      }
    });
    
    // Definite references (the + noun)
    const definitePattern = /\bthe\s+([a-z]+(?:\s+[a-z]+)*)/gi;
    let match;
    while ((match = definitePattern.exec(content)) !== null) {
      references.push(this.createReference(
        match[0],
        'definite',
        timestamp,
        spatialInfo
      ));
    }
    
    return references;
  }

  // Create a new entity
  private createEntity(name: string, type: Entity['type'], timestamp: number): Entity {
    const entityId = `entity_${name.replace(/\s+/g, '_').toLowerCase()}_${timestamp}`;
    
    return {
      id: entityId,
      name,
      type,
      aliases: [name],
      firstMention: timestamp,
      lastMention: timestamp,
      mentionCount: 1,
      confidence: 0.8,
      attributes: new Map(),
      relationships: new Map()
    };
  }

  // Create a new reference
  private createReference(
    text: string,
    type: Reference['type'],
    timestamp: number,
    spatialInfo?: Map<string, [number, number, number]>
  ): Reference {
    const referenceId = `ref_${timestamp}_${Math.random().toString(36).substr(2, 9)}`;
    
    return {
      id: referenceId,
      entityId: '', // Will be resolved later
      text,
      type,
      timestamp,
      confidence: 0.7
    };
  }

  // Add or update entity
  private addOrUpdateEntity(entity: Entity): void {
    const existingEntity = this.findExistingEntity(entity.name, entity.type);
    
    if (existingEntity) {
      // Update existing entity
      existingEntity.lastMention = entity.lastMention;
      existingEntity.mentionCount++;
      existingEntity.confidence = Math.min(1.0, existingEntity.confidence + 0.1);
      
      // Add new aliases
      if (!existingEntity.aliases.includes(entity.name)) {
        existingEntity.aliases.push(entity.name);
      }
    } else {
      // Add new entity
      this.entities.set(entity.id, entity);
    }
  }

  // Find existing entity by name and type
  private findExistingEntity(name: string, type: Entity['type']): Entity | null {
    for (const entity of this.entities.values()) {
      if (entity.type === type && 
          (entity.name.toLowerCase() === name.toLowerCase() ||
           entity.aliases.some(alias => alias.toLowerCase() === name.toLowerCase()))) {
        return entity;
      }
    }
    return null;
  }

  // Add reference
  private addReference(reference: Reference): void {
    this.references.set(reference.id, reference);
  }

  // Resolve reference by spatial location
  private resolveSpatialReference(location: [number, number, number]): Entity | null {
    const threshold = 0.2; // Spatial proximity threshold
    
    for (const [entityId, entityLocation] of this.spatialMemory) {
      const distance = Math.sqrt(
        Math.pow(location[0] - entityLocation[0], 2) +
        Math.pow(location[1] - entityLocation[1], 2) +
        Math.pow(location[2] - entityLocation[2], 2)
      );
      
      if (distance < threshold) {
        return this.entities.get(entityId) || null;
      }
    }
    
    return null;
  }

  // Resolve reference by recency
  private resolveByRecency(referenceText: string, timestamp: number): Entity | null {
    const recentWindow = 30000; // 30 seconds
    const recentEntities = Array.from(this.entities.values())
      .filter(entity => timestamp - entity.lastMention < recentWindow)
      .sort((a, b) => b.lastMention - a.lastMention);
    
    // Simple heuristics for pronoun resolution
    if (['he', 'him', 'his'].includes(referenceText.toLowerCase())) {
      return recentEntities.find(entity => entity.type === 'person') || null;
    }
    
    if (['she', 'her', 'hers'].includes(referenceText.toLowerCase())) {
      return recentEntities.find(entity => entity.type === 'person') || null;
    }
    
    if (['it', 'its'].includes(referenceText.toLowerCase())) {
      return recentEntities.find(entity => entity.type !== 'person') || null;
    }
    
    if (['they', 'them', 'their'].includes(referenceText.toLowerCase())) {
      return recentEntities[0] || null; // Most recent entity
    }
    
    return null;
  }

  // Resolve reference by salience
  private resolveBySalience(referenceText: string): Entity | null {
    const entitiesBySalience = Array.from(this.entities.values())
      .sort((a, b) => {
        // Salience score based on mention count and recency
        const aScore = a.mentionCount * Math.exp(-(Date.now() - a.lastMention) / 60000);
        const bScore = b.mentionCount * Math.exp(-(Date.now() - b.lastMention) / 60000);
        return bScore - aScore;
      });
    
    return entitiesBySalience[0] || null;
  }

  // Resolve reference syntactically
  private resolveSyntactically(referenceText: string, timestamp: number): Entity | null {
    // Find the most recent turn and look for syntactic cues
    const recentTurn = this.turns[this.turns.length - 1];
    if (!recentTurn) return null;
    
    // Simple subject-object tracking
    const sentences = recentTurn.content.split(/[.!?]/);
    const lastSentence = sentences[sentences.length - 1];
    
    // Extract potential antecedents
    const words = lastSentence.split(/\s+/);
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const entity = this.findExistingEntity(word, 'person');
      if (entity) {
        return entity;
      }
    }
    
    return null;
  }

  // Update topic segmentation
  private updateTopicSegmentation(turn: ConversationTurn): void {
    const currentTopic = this.topics[this.topics.length - 1];
    
    if (!currentTopic) {
      // Start first topic
      this.topics.push({
        id: `topic_${turn.startTime}`,
        startTime: turn.startTime,
        mainTopic: this.extractMainTopic(turn.content),
        subTopics: [],
        entities: turn.entities,
        coherenceScore: 1.0
      });
      return;
    }
    
    // Calculate topic coherence with current segment
    const coherence = this.calculateTopicCoherence(turn, currentTopic);
    
    if (coherence > 0.6) {
      // Continue current topic
      currentTopic.entities = [...new Set([...currentTopic.entities, ...turn.entities])];
      currentTopic.coherenceScore = (currentTopic.coherenceScore + coherence) / 2;
    } else {
      // Start new topic segment
      currentTopic.endTime = turn.startTime;
      
      this.topics.push({
        id: `topic_${turn.startTime}`,
        startTime: turn.startTime,
        mainTopic: this.extractMainTopic(turn.content),
        subTopics: [],
        entities: turn.entities,
        coherenceScore: 1.0,
        transitionType: this.detectTransitionType(currentTopic, turn)
      });
    }
  }

  // Extract main topic from content
  private extractMainTopic(content: string): string {
    // Simple keyword extraction (in production, use topic modeling)
    const words = content.toLowerCase().split(/\s+/);
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']);
    
    const keywords = words
      .filter(word => word.length > 3 && !stopWords.has(word))
      .reduce((acc, word) => {
        acc[word] = (acc[word] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
    
    const topKeyword = Object.entries(keywords)
      .sort(([, a], [, b]) => b - a)[0];
    
    return topKeyword ? topKeyword[0] : 'general';
  }

  // Calculate topic coherence
  private calculateTopicCoherence(turn: ConversationTurn, topic: TopicSegment): number {
    // Entity overlap
    const entityOverlap = turn.entities.filter(e => topic.entities.includes(e)).length;
    const entityCoherence = entityOverlap / Math.max(turn.entities.length, topic.entities.length, 1);
    
    // Lexical similarity (simplified)
    const turnWords = new Set(turn.content.toLowerCase().split(/\s+/));
    const topicWords = new Set(topic.mainTopic.toLowerCase().split(/\s+/));
    const wordOverlap = [...turnWords].filter(w => topicWords.has(w)).length;
    const lexicalCoherence = wordOverlap / Math.max(turnWords.size, topicWords.size, 1);
    
    return (entityCoherence + lexicalCoherence) / 2;
  }

  // Detect topic transition type
  private detectTransitionType(previousTopic: TopicSegment, turn: ConversationTurn): TopicSegment['transitionType'] {
    // Check if returning to a previous topic
    const previousTopics = this.topics.slice(0, -1);
    const mainTopic = this.extractMainTopic(turn.content);
    
    for (const topic of previousTopics.reverse()) {
      if (topic.mainTopic === mainTopic) {
        return 'return';
      }
    }
    
    // Check for explicit transition markers
    const transitionMarkers = {
      shift: ['anyway', 'moving on', 'speaking of', 'by the way'],
      digression: ['oh', 'wait', 'actually', 'before I forget'],
      continuation: ['also', 'and', 'furthermore', 'moreover']
    };
    
    const content = turn.content.toLowerCase();
    
    for (const [type, markers] of Object.entries(transitionMarkers)) {
      if (markers.some(marker => content.includes(marker))) {
        return type as TopicSegment['transitionType'];
      }
    }
    
    return 'shift';
  }

  // Prune memory to maintain size limits
  private pruneMemory(): void {
    if (this.entities.size > this.maxMemorySize) {
      // Remove least recently mentioned entities
      const sortedEntities = Array.from(this.entities.entries())
        .sort(([, a], [, b]) => a.lastMention - b.lastMention);
      
      const toRemove = sortedEntities.slice(0, sortedEntities.length - this.maxMemorySize);
      toRemove.forEach(([id]) => {
        this.entities.delete(id);
        this.spatialMemory.delete(id);
      });
    }
    
    // Prune old turns
    if (this.turns.length > this.maxMemorySize) {
      this.turns = this.turns.slice(-this.maxMemorySize);
    }
    
    // Prune old references
    if (this.references.size > this.maxMemorySize) {
      const sortedRefs = Array.from(this.references.entries())
        .sort(([, a], [, b]) => a.timestamp - b.timestamp);
      
      const toRemove = sortedRefs.slice(0, sortedRefs.length - this.maxMemorySize);
      toRemove.forEach(([id]) => this.references.delete(id));
    }
  }

  // Export memory state
  public exportMemory(): any {
    return {
      conversationId: this.conversationId,
      entities: Array.from(this.entities.entries()),
      references: Array.from(this.references.entries()),
      turns: this.turns,
      topics: this.topics,
      spatialMemory: Array.from(this.spatialMemory.entries()),
      temporalContext: Array.from(this.temporalContext.entries())
    };
  }

  // Import memory state
  public importMemory(data: any): void {
    this.conversationId = data.conversationId;
    this.entities = new Map(data.entities);
    this.references = new Map(data.references);
    this.turns = data.turns;
    this.topics = data.topics;
    this.spatialMemory = new Map(data.spatialMemory);
    this.temporalContext = new Map(data.temporalContext);
  }

  // Clear all memory
  public clearMemory(): void {
    this.entities.clear();
    this.references.clear();
    this.turns = [];
    this.topics = [];
    this.spatialMemory.clear();
    this.temporalContext.clear();
  }
}
