// Text-to-Sign Pipeline for reverse mode translation

interface SignGrammarRule {
  pattern: RegExp;
  glossSequence: string[];
  handshapes: string[];
  movements: string[];
  nonManualMarkers: string[];
}

interface SignPose {
  timestamp: number;
  duration: number;
  gloss: string;
  handShape: {
    left: string;
    right: string;
  };
  movement: {
    type: string;
    direction: [number, number, number];
    speed: number;
  };
  location: {
    x: number;
    y: number;
    z: number;
  };
  nonManualMarkers: {
    eyebrows: 'raised' | 'furrowed' | 'neutral';
    eyeGaze: 'forward' | 'left' | 'right' | 'up' | 'down';
    mouthShape: string;
    headMovement: 'nod' | 'shake' | 'tilt' | 'none';
  };
}

interface SignSequence {
  poses: SignPose[];
  totalDuration: number;
  confidence: number;
  metadata: {
    originalText: string;
    glossSequence: string[];
    complexity: 'simple' | 'moderate' | 'complex';
  };
}

export class TextToSignPipeline {
  private grammarRules: SignGrammarRule[];
  private vocabulary: Map<string, SignPose>;
  
  constructor() {
    this.grammarRules = this.initializeGrammarRules();
    this.vocabulary = this.initializeVocabulary();
  }

  // Main pipeline: convert text to sign sequence
  public async convertTextToSigns(text: string): Promise<SignSequence> {
    try {
      // Step 1: Text preprocessing
      const preprocessedText = this.preprocessText(text);
      
      // Step 2: Linguistic analysis
      const linguisticAnalysis = this.analyzeLinguistics(preprocessedText);
      
      // Step 3: Grammar planning
      const grammarPlan = this.planSignGrammar(linguisticAnalysis);
      
      // Step 4: Gloss generation
      const glossSequence = this.generateGlossSequence(grammarPlan);
      
      // Step 5: Pose generation
      const poses = this.generateSignPoses(glossSequence);
      
      // Step 6: Temporal alignment
      const alignedPoses = this.alignTemporal(poses);
      
      // Step 7: Smoothing and transitions
      const smoothedPoses = this.applySmoothingTransitions(alignedPoses);
      
      return {
        poses: smoothedPoses,
        totalDuration: this.calculateTotalDuration(smoothedPoses),
        confidence: this.calculateConfidence(text, smoothedPoses),
        metadata: {
          originalText: text,
          glossSequence: glossSequence,
          complexity: this.assessComplexity(text)
        }
      };
      
    } catch (error) {
      console.error('Error in text-to-sign conversion:', error);
      throw new Error(`Failed to convert text to signs: ${error}`);
    }
  }

  // Step 1: Preprocess input text
  private preprocessText(text: string): string {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s\?\!\.\,]/g, '') // Remove special characters except punctuation
      .replace(/\s+/g, ' '); // Normalize whitespace
  }

  // Step 2: Analyze linguistic structure
  private analyzeLinguistics(text: string): any {
    const words = text.split(' ');
    
    return {
      words: words,
      wordCount: words.length,
      hasQuestion: text.includes('?'),
      hasExclamation: text.includes('!'),
      tense: this.detectTense(text),
      entities: this.extractEntities(text),
      sentiment: this.analyzeSentiment(text)
    };
  }

  // Detect grammatical tense
  private detectTense(text: string): 'past' | 'present' | 'future' {
    if (text.includes('will') || text.includes('going to')) return 'future';
    if (text.includes('was') || text.includes('were') || text.includes('ed ')) return 'past';
    return 'present';
  }

  // Extract named entities
  private extractEntities(text: string): string[] {
    // Simple entity extraction - in production would use NLP library
    const entities = [];
    const words = text.split(' ');
    
    for (const word of words) {
      if (word[0] === word[0].toUpperCase() && word.length > 1) {
        entities.push(word);
      }
    }
    
    return entities;
  }

  // Analyze sentiment
  private analyzeSentiment(text: string): 'positive' | 'negative' | 'neutral' {
    const positiveWords = ['good', 'great', 'happy', 'love', 'like', 'wonderful', 'excellent'];
    const negativeWords = ['bad', 'sad', 'hate', 'terrible', 'awful', 'horrible'];
    
    const words = text.split(' ');
    let score = 0;
    
    for (const word of words) {
      if (positiveWords.includes(word)) score++;
      if (negativeWords.includes(word)) score--;
    }
    
    if (score > 0) return 'positive';
    if (score < 0) return 'negative';
    return 'neutral';
  }

  // Step 3: Plan sign language grammar
  private planSignGrammar(analysis: any): any {
    const plan = {
      topicalization: this.planTopicalization(analysis),
      spatialSetup: this.planSpatialSetup(analysis),
      timeReference: this.planTimeReference(analysis),
      nonManualMarkers: this.planNonManualMarkers(analysis)
    };
    
    return plan;
  }

  // Plan topic-comment structure
  private planTopicalization(analysis: any): any {
    // ASL often uses topic-comment structure
    if (analysis.hasQuestion) {
      return { type: 'question', structure: 'wh-question' };
    }
    
    return { type: 'statement', structure: 'topic-comment' };
  }

  // Plan spatial reference setup
  private planSpatialSetup(analysis: any): any {
    const entities = analysis.entities;
    const spatialMap = new Map();
    
    // Assign spatial locations to entities
    entities.forEach((entity: string, index: number) => {
      const locations = [
        { x: -0.3, y: 0.0, z: 0.0 }, // Left
        { x: 0.3, y: 0.0, z: 0.0 },  // Right
        { x: 0.0, y: -0.2, z: 0.0 }, // Upper
        { x: 0.0, y: 0.2, z: 0.0 }   // Lower
      ];
      
      spatialMap.set(entity, locations[index % locations.length]);
    });
    
    return { entityLocations: spatialMap };
  }

  // Plan time reference markers
  private planTimeReference(analysis: any): any {
    const tenseMarkers = {
      past: { location: 'behind', movement: 'backward' },
      present: { location: 'center', movement: 'neutral' },
      future: { location: 'forward', movement: 'forward' }
    };
    
    return tenseMarkers[analysis.tense];
  }

  // Plan non-manual markers
  private planNonManualMarkers(analysis: any): any {
    const markers = {
      eyebrows: 'neutral' as const,
      eyeGaze: 'forward' as const,
      mouthShape: 'neutral',
      headMovement: 'none' as const
    };
    
    if (analysis.hasQuestion) {
      markers.eyebrows = 'raised';
      markers.headMovement = 'tilt';
    }
    
    if (analysis.sentiment === 'positive') {
      markers.mouthShape = 'smile';
    } else if (analysis.sentiment === 'negative') {
      markers.eyebrows = 'furrowed';
    }
    
    return markers;
  }

  // Step 4: Generate gloss sequence
  private generateGlossSequence(grammarPlan: any): string[] {
    // This would use the grammar plan to generate appropriate gloss sequence
    // For now, return a simplified mapping
    return ['HELLO', 'YOU', 'HOW', 'YOU'];
  }

  // Step 5: Generate sign poses from gloss sequence
  private generateSignPoses(glossSequence: string[]): SignPose[] {
    const poses: SignPose[] = [];
    let currentTime = 0;
    
    for (const gloss of glossSequence) {
      const basePose = this.vocabulary.get(gloss) || this.getDefaultPose(gloss);
      
      const pose: SignPose = {
        ...basePose,
        timestamp: currentTime,
        gloss: gloss
      };
      
      poses.push(pose);
      currentTime += pose.duration;
    }
    
    return poses;
  }

  // Get default pose for unknown gloss
  private getDefaultPose(gloss: string): SignPose {
    return {
      timestamp: 0,
      duration: 800, // 800ms default
      gloss: gloss,
      handShape: {
        left: 'flat',
        right: 'flat'
      },
      movement: {
        type: 'static',
        direction: [0, 0, 0],
        speed: 1.0
      },
      location: {
        x: 0,
        y: 0,
        z: 0
      },
      nonManualMarkers: {
        eyebrows: 'neutral',
        eyeGaze: 'forward',
        mouthShape: 'neutral',
        headMovement: 'none'
      }
    };
  }

  // Step 6: Temporal alignment
  private alignTemporal(poses: SignPose[]): SignPose[] {
    // Adjust timing for natural flow
    const alignedPoses = [...poses];
    
    for (let i = 0; i < alignedPoses.length; i++) {
      const pose = alignedPoses[i];
      
      // Add transition time between poses
      if (i > 0) {
        const transitionTime = 100; // 100ms transition
        pose.timestamp += transitionTime;
      }
      
      // Adjust duration based on complexity
      if (pose.movement.type !== 'static') {
        pose.duration *= 1.2; // Longer for movement
      }
    }
    
    return alignedPoses;
  }

  // Step 7: Apply smoothing and transitions
  private applySmoothingTransitions(poses: SignPose[]): SignPose[] {
    // Add transition poses between main poses
    const smoothedPoses: SignPose[] = [];
    
    for (let i = 0; i < poses.length; i++) {
      smoothedPoses.push(poses[i]);
      
      // Add transition to next pose
      if (i < poses.length - 1) {
        const transitionPose = this.createTransitionPose(poses[i], poses[i + 1]);
        smoothedPoses.push(transitionPose);
      }
    }
    
    return smoothedPoses;
  }

  // Create transition pose between two poses
  private createTransitionPose(fromPose: SignPose, toPose: SignPose): SignPose {
    return {
      timestamp: fromPose.timestamp + fromPose.duration,
      duration: 150, // 150ms transition
      gloss: 'TRANSITION',
      handShape: {
        left: 'transitional',
        right: 'transitional'
      },
      movement: {
        type: 'transition',
        direction: [
          (toPose.location.x - fromPose.location.x) / 2,
          (toPose.location.y - fromPose.location.y) / 2,
          (toPose.location.z - fromPose.location.z) / 2
        ],
        speed: 1.5
      },
      location: {
        x: (fromPose.location.x + toPose.location.x) / 2,
        y: (fromPose.location.y + toPose.location.y) / 2,
        z: (fromPose.location.z + toPose.location.z) / 2
      },
      nonManualMarkers: fromPose.nonManualMarkers
    };
  }

  // Calculate total duration
  private calculateTotalDuration(poses: SignPose[]): number {
    if (poses.length === 0) return 0;
    
    const lastPose = poses[poses.length - 1];
    return lastPose.timestamp + lastPose.duration;
  }

  // Calculate confidence score
  private calculateConfidence(originalText: string, poses: SignPose[]): number {
    // Base confidence on vocabulary coverage and complexity
    const words = originalText.split(' ');
    const knownWords = words.filter(word => this.vocabulary.has(word.toUpperCase()));
    
    const vocabularyCoverage = knownWords.length / words.length;
    const complexityPenalty = this.assessComplexity(originalText) === 'complex' ? 0.1 : 0;
    
    return Math.max(0.3, Math.min(1.0, vocabularyCoverage - complexityPenalty));
  }

  // Assess text complexity
  private assessComplexity(text: string): 'simple' | 'moderate' | 'complex' {
    const words = text.split(' ');
    
    if (words.length <= 3) return 'simple';
    if (words.length <= 8) return 'moderate';
    return 'complex';
  }

  // Initialize grammar rules
  private initializeGrammarRules(): SignGrammarRule[] {
    return [
      {
        pattern: /^(hello|hi|hey)/i,
        glossSequence: ['HELLO'],
        handshapes: ['flat'],
        movements: ['wave'],
        nonManualMarkers: ['smile']
      },
      {
        pattern: /thank you/i,
        glossSequence: ['THANK', 'YOU'],
        handshapes: ['flat', 'point'],
        movements: ['forward', 'point'],
        nonManualMarkers: ['smile']
      }
      // Add more rules as needed
    ];
  }

  // Initialize vocabulary
  private initializeVocabulary(): Map<string, SignPose> {
    const vocab = new Map<string, SignPose>();
    
    // Add common signs
    vocab.set('HELLO', {
      timestamp: 0,
      duration: 800,
      gloss: 'HELLO',
      handShape: { left: 'flat', right: 'flat' },
      movement: { type: 'wave', direction: [1, 0, 0], speed: 1.0 },
      location: { x: 0.3, y: -0.2, z: 0 },
      nonManualMarkers: {
        eyebrows: 'neutral',
        eyeGaze: 'forward',
        mouthShape: 'smile',
        headMovement: 'none'
      }
    });
    
    vocab.set('THANK', {
      timestamp: 0,
      duration: 600,
      gloss: 'THANK',
      handShape: { left: 'flat', right: 'flat' },
      movement: { type: 'forward', direction: [0, 1, 0], speed: 1.2 },
      location: { x: 0, y: -0.1, z: 0 },
      nonManualMarkers: {
        eyebrows: 'neutral',
        eyeGaze: 'forward',
        mouthShape: 'neutral',
        headMovement: 'nod'
      }
    });
    
    vocab.set('YOU', {
      timestamp: 0,
      duration: 500,
      gloss: 'YOU',
      handShape: { left: 'neutral', right: 'point' },
      movement: { type: 'point', direction: [0, 0, 1], speed: 1.0 },
      location: { x: 0.2, y: 0, z: 0 },
      nonManualMarkers: {
        eyebrows: 'neutral',
        eyeGaze: 'forward',
        mouthShape: 'neutral',
        headMovement: 'none'
      }
    });
    
    return vocab;
  }
}
