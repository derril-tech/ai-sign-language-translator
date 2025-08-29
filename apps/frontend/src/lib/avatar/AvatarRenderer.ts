// 3D Avatar Renderer using Three.js
// This is a simplified implementation - in production would use full Three.js with GLTF models

interface AvatarJoint {
  name: string;
  position: [number, number, number];
  rotation: [number, number, number];
}

interface AvatarPose {
  timestamp: number;
  joints: AvatarJoint[];
  confidence: number;
}

export class AvatarRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animationFrame: number | null = null;
  private currentPose: AvatarPose | null = null;
  private isPlaying = false;

  // Avatar skeleton structure
  private skeleton = {
    head: { x: 0, y: -60, radius: 40 },
    neck: { x: 0, y: -20 },
    torso: { x: 0, y: 0, width: 60, height: 80 },
    leftShoulder: { x: -30, y: -10 },
    rightShoulder: { x: 30, y: -10 },
    leftElbow: { x: -50, y: 20 },
    rightElbow: { x: 50, y: 20 },
    leftHand: { x: -70, y: 50, radius: 8 },
    rightHand: { x: 70, y: 50, radius: 8 },
    leftHip: { x: -20, y: 60 },
    rightHip: { x: 20, y: 60 }
  };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Could not get 2D context from canvas');
    }
    this.ctx = context;
    
    // Set canvas size
    this.canvas.width = 300;
    this.canvas.height = 400;
  }

  // Initialize the avatar renderer
  public initialize(): void {
    this.drawDefaultPose();
  }

  // Set the current pose for the avatar
  public setPose(pose: AvatarPose): void {
    this.currentPose = pose;
    this.render();
  }

  // Start animation playback
  public play(): void {
    this.isPlaying = true;
    this.animate();
  }

  // Stop animation playback
  public stop(): void {
    this.isPlaying = false;
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  // Animate text to sign language
  public animateText(text: string): void {
    // Convert text to sign sequence
    const signSequence = this.textToSignSequence(text);
    this.playSignSequence(signSequence);
  }

  // Convert text to sign language sequence
  private textToSignSequence(text: string): AvatarPose[] {
    const words = text.toLowerCase().split(' ');
    const poses: AvatarPose[] = [];
    
    words.forEach((word, index) => {
      const pose = this.getSignPoseForWord(word, index);
      poses.push(pose);
    });

    return poses;
  }

  // Get sign pose for a specific word
  private getSignPoseForWord(word: string, index: number): AvatarPose {
    // Mock sign poses for common words
    const signPoses: { [key: string]: Partial<typeof this.skeleton> } = {
      'hello': {
        rightHand: { x: 90, y: -20, radius: 8 },
        rightElbow: { x: 60, y: 0 },
        leftHand: { x: -50, y: 30, radius: 8 }
      },
      'thank': {
        rightHand: { x: 40, y: -30, radius: 8 },
        leftHand: { x: -40, y: -30, radius: 8 },
        rightElbow: { x: 30, y: -10 },
        leftElbow: { x: -30, y: -10 }
      },
      'you': {
        rightHand: { x: 80, y: -10, radius: 8 },
        rightElbow: { x: 50, y: 10 }
      },
      'please': {
        rightHand: { x: 20, y: -10, radius: 8 },
        rightElbow: { x: 40, y: 10 }
      },
      'good': {
        rightHand: { x: 60, y: -40, radius: 8 },
        rightElbow: { x: 40, y: -20 }
      },
      'yes': {
        rightHand: { x: 70, y: -30, radius: 8 },
        rightElbow: { x: 50, y: -10 }
      },
      'no': {
        rightHand: { x: 80, y: 0, radius: 8 },
        leftHand: { x: -80, y: 0, radius: 8 },
        rightElbow: { x: 60, y: 20 },
        leftElbow: { x: -60, y: 20 }
      }
    };

    const poseAdjustments = signPoses[word] || {};
    
    return {
      timestamp: Date.now() + index * 1000,
      joints: this.skeletonToJoints({ ...this.skeleton, ...poseAdjustments }),
      confidence: 0.8 + Math.random() * 0.2
    };
  }

  // Convert skeleton to joints array
  private skeletonToJoints(skeleton: typeof this.skeleton): AvatarJoint[] {
    return Object.entries(skeleton).map(([name, data]) => ({
      name,
      position: [data.x || 0, data.y || 0, 0],
      rotation: [0, 0, 0]
    }));
  }

  // Play a sequence of sign poses
  private playSignSequence(poses: AvatarPose[]): void {
    let currentIndex = 0;
    
    const playNext = () => {
      if (currentIndex < poses.length && this.isPlaying) {
        this.setPose(poses[currentIndex]);
        currentIndex++;
        setTimeout(playNext, 800); // 800ms per sign
      } else {
        this.isPlaying = false;
      }
    };

    this.isPlaying = true;
    playNext();
  }

  // Main animation loop
  private animate(): void {
    if (!this.isPlaying) return;

    this.render();
    this.animationFrame = requestAnimationFrame(() => this.animate());
  }

  // Render the current avatar state
  private render(): void {
    const centerX = this.canvas.width / 2;
    const centerY = this.canvas.height / 2;

    // Clear canvas
    this.ctx.fillStyle = '#1a1a2e';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Use current pose or default
    const pose = this.currentPose || {
      timestamp: Date.now(),
      joints: this.skeletonToJoints(this.skeleton),
      confidence: 1.0
    };

    // Draw avatar
    this.drawAvatar(centerX, centerY, pose);
    
    // Draw confidence indicator
    this.drawConfidenceIndicator(pose.confidence);
  }

  // Draw the avatar
  private drawAvatar(centerX: number, centerY: number, pose: AvatarPose): void {
    const joints = this.jointsToSkeleton(pose.joints);

    // Draw connections (bones)
    this.ctx.strokeStyle = '#4a90e2';
    this.ctx.lineWidth = 4;
    this.ctx.lineCap = 'round';

    const connections = [
      ['head', 'neck'],
      ['neck', 'torso'],
      ['neck', 'leftShoulder'],
      ['neck', 'rightShoulder'],
      ['leftShoulder', 'leftElbow'],
      ['rightShoulder', 'rightElbow'],
      ['leftElbow', 'leftHand'],
      ['rightElbow', 'rightHand'],
      ['torso', 'leftHip'],
      ['torso', 'rightHip']
    ];

    connections.forEach(([start, end]) => {
      const startJoint = joints[start];
      const endJoint = joints[end];
      
      if (startJoint && endJoint) {
        this.ctx.beginPath();
        this.ctx.moveTo(centerX + startJoint.x, centerY + startJoint.y);
        this.ctx.lineTo(centerX + endJoint.x, centerY + endJoint.y);
        this.ctx.stroke();
      }
    });

    // Draw head
    this.ctx.fillStyle = '#fdbcb4';
    this.ctx.beginPath();
    this.ctx.arc(centerX + joints.head.x, centerY + joints.head.y, joints.head.radius, 0, 2 * Math.PI);
    this.ctx.fill();

    // Draw facial features
    this.drawFace(centerX + joints.head.x, centerY + joints.head.y);

    // Draw torso
    this.ctx.fillStyle = '#4a90e2';
    this.ctx.fillRect(
      centerX + joints.torso.x - joints.torso.width / 2,
      centerY + joints.torso.y,
      joints.torso.width,
      joints.torso.height
    );

    // Draw hands
    this.ctx.fillStyle = '#fdbcb4';
    
    // Left hand
    this.ctx.beginPath();
    this.ctx.arc(centerX + joints.leftHand.x, centerY + joints.leftHand.y, joints.leftHand.radius, 0, 2 * Math.PI);
    this.ctx.fill();

    // Right hand
    this.ctx.beginPath();
    this.ctx.arc(centerX + joints.rightHand.x, centerY + joints.rightHand.y, joints.rightHand.radius, 0, 2 * Math.PI);
    this.ctx.fill();

    // Draw hand indicators for signing
    if (this.isPlaying) {
      this.drawHandIndicators(centerX, centerY, joints);
    }
  }

  // Draw facial features
  private drawFace(centerX: number, centerY: number): void {
    // Eyes
    this.ctx.fillStyle = '#333';
    this.ctx.beginPath();
    this.ctx.arc(centerX - 12, centerY - 5, 3, 0, 2 * Math.PI);
    this.ctx.arc(centerX + 12, centerY - 5, 3, 0, 2 * Math.PI);
    this.ctx.fill();

    // Mouth
    this.ctx.strokeStyle = '#333';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY + 10, 8, 0, Math.PI);
    this.ctx.stroke();
  }

  // Draw hand movement indicators
  private drawHandIndicators(centerX: number, centerY: number, joints: any): void {
    const time = Date.now() * 0.005;
    
    // Left hand trail
    this.ctx.strokeStyle = '#ff6b6b';
    this.ctx.lineWidth = 2;
    this.ctx.globalAlpha = 0.6;
    
    for (let i = 0; i < 5; i++) {
      const offset = i * 5;
      const alpha = 1 - (i * 0.2);
      this.ctx.globalAlpha = alpha * 0.6;
      
      this.ctx.beginPath();
      this.ctx.arc(
        centerX + joints.leftHand.x - offset,
        centerY + joints.leftHand.y,
        joints.leftHand.radius - i,
        0, 2 * Math.PI
      );
      this.ctx.stroke();
    }

    // Right hand trail
    this.ctx.strokeStyle = '#4ecdc4';
    
    for (let i = 0; i < 5; i++) {
      const offset = i * 5;
      const alpha = 1 - (i * 0.2);
      this.ctx.globalAlpha = alpha * 0.6;
      
      this.ctx.beginPath();
      this.ctx.arc(
        centerX + joints.rightHand.x + offset,
        centerY + joints.rightHand.y,
        joints.rightHand.radius - i,
        0, 2 * Math.PI
      );
      this.ctx.stroke();
    }

    this.ctx.globalAlpha = 1;
  }

  // Draw confidence indicator
  private drawConfidenceIndicator(confidence: number): void {
    const barWidth = 100;
    const barHeight = 8;
    const x = 10;
    const y = this.canvas.height - 30;

    // Background
    this.ctx.fillStyle = '#333';
    this.ctx.fillRect(x, y, barWidth, barHeight);

    // Confidence bar
    const color = confidence >= 0.8 ? '#10b981' : confidence >= 0.6 ? '#f59e0b' : '#ef4444';
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x, y, barWidth * confidence, barHeight);

    // Text
    this.ctx.fillStyle = '#fff';
    this.ctx.font = '12px Arial';
    this.ctx.fillText(`${Math.round(confidence * 100)}%`, x + barWidth + 10, y + 6);
  }

  // Convert joints array back to skeleton object
  private jointsToSkeleton(joints: AvatarJoint[]): any {
    const skeleton = { ...this.skeleton };
    
    joints.forEach(joint => {
      if (skeleton[joint.name as keyof typeof skeleton]) {
        (skeleton as any)[joint.name].x = joint.position[0];
        (skeleton as any)[joint.name].y = joint.position[1];
      }
    });

    return skeleton;
  }

  // Draw default pose
  private drawDefaultPose(): void {
    this.currentPose = {
      timestamp: Date.now(),
      joints: this.skeletonToJoints(this.skeleton),
      confidence: 1.0
    };
    this.render();
  }

  // Cleanup
  public dispose(): void {
    this.stop();
  }
}
