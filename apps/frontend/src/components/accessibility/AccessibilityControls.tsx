'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Type, 
  Eye, 
  Ear, 
  Palette, 
  Zap, 
  Volume2, 
  MousePointer,
  Keyboard,
  Monitor,
  Settings,
  RotateCcw,
  Save
} from 'lucide-react';

interface AccessibilitySettings {
  // Visual
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  fontFamily: 'default' | 'dyslexia-friendly' | 'high-contrast';
  colorScheme: 'default' | 'high-contrast' | 'dark' | 'light' | 'protanopia' | 'deuteranopia' | 'tritanopia';
  
  // Captions
  captionSize: 'small' | 'medium' | 'large' | 'extra-large';
  captionBackground: 'transparent' | 'semi-transparent' | 'solid';
  captionPosition: 'bottom' | 'top' | 'overlay';
  showConfidence: boolean;
  showTimestamps: boolean;
  
  // Audio
  audioDescriptions: boolean;
  soundEffects: boolean;
  voiceAnnouncements: boolean;
  
  // Motor
  reduceMotion: boolean;
  largerClickTargets: boolean;
  stickyHover: boolean;
  keyboardNavigation: boolean;
  
  // Cognitive
  simplifiedInterface: boolean;
  focusIndicators: boolean;
  autoSave: boolean;
  confirmActions: boolean;
  
  // Screen Reader
  screenReaderOptimized: boolean;
  ariaLabels: boolean;
  skipLinks: boolean;
}

interface AccessibilityControlsProps {
  settings: AccessibilitySettings;
  onSettingsChange: (settings: AccessibilitySettings) => void;
  className?: string;
}

export function AccessibilityControls({
  settings,
  onSettingsChange,
  className = ''
}: AccessibilityControlsProps) {
  const [activeTab, setActiveTab] = useState<'visual' | 'audio' | 'motor' | 'cognitive'>('visual');
  const [showPreview, setShowPreview] = useState(false);

  // Apply accessibility settings to document
  useEffect(() => {
    applyAccessibilitySettings(settings);
  }, [settings]);

  // Handle setting changes
  const updateSetting = useCallback(<K extends keyof AccessibilitySettings>(
    key: K,
    value: AccessibilitySettings[K]
  ) => {
    const newSettings = { ...settings, [key]: value };
    onSettingsChange(newSettings);
  }, [settings, onSettingsChange]);

  // Reset to defaults
  const resetToDefaults = useCallback(() => {
    const defaultSettings: AccessibilitySettings = {
      fontSize: 'medium',
      fontFamily: 'default',
      colorScheme: 'default',
      captionSize: 'medium',
      captionBackground: 'semi-transparent',
      captionPosition: 'bottom',
      showConfidence: false,
      showTimestamps: false,
      audioDescriptions: false,
      soundEffects: true,
      voiceAnnouncements: false,
      reduceMotion: false,
      largerClickTargets: false,
      stickyHover: false,
      keyboardNavigation: true,
      simplifiedInterface: false,
      focusIndicators: true,
      autoSave: true,
      confirmActions: false,
      screenReaderOptimized: false,
      ariaLabels: true,
      skipLinks: true
    };
    onSettingsChange(defaultSettings);
  }, [onSettingsChange]);

  // Save settings to localStorage
  const saveSettings = useCallback(() => {
    localStorage.setItem('accessibility-settings', JSON.stringify(settings));
  }, [settings]);

  const tabs = [
    { id: 'visual' as const, label: 'Visual', icon: Eye },
    { id: 'audio' as const, label: 'Audio', icon: Ear },
    { id: 'motor' as const, label: 'Motor', icon: MousePointer },
    { id: 'cognitive' as const, label: 'Cognitive', icon: Monitor }
  ];

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Settings className="w-5 h-5 text-blue-400" />
            <h3 className="text-white font-medium">Accessibility Settings</h3>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowPreview(!showPreview)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Preview changes"
            >
              <Eye className="w-4 h-4" />
            </button>
            
            <button
              onClick={resetToDefaults}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Reset to defaults"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            
            <button
              onClick={saveSettings}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Save settings"
            >
              <Save className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-black/20 rounded-lg p-1">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  relative flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all
                  ${activeTab === tab.id
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-300 hover:text-white hover:bg-white/10'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
                
                {activeTab === tab.id && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-blue-500 rounded-md -z-10"
                    initial={false}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content */}
      <div className="p-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {activeTab === 'visual' && (
              <VisualSettings settings={settings} updateSetting={updateSetting} />
            )}
            {activeTab === 'audio' && (
              <AudioSettings settings={settings} updateSetting={updateSetting} />
            )}
            {activeTab === 'motor' && (
              <MotorSettings settings={settings} updateSetting={updateSetting} />
            )}
            {activeTab === 'cognitive' && (
              <CognitiveSettings settings={settings} updateSetting={updateSetting} />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Preview Panel */}
      <AnimatePresence>
        {showPreview && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-white/10 p-4"
          >
            <AccessibilityPreview settings={settings} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Visual Settings Component
function VisualSettings({ 
  settings, 
  updateSetting 
}: { 
  settings: AccessibilitySettings; 
  updateSetting: (key: keyof AccessibilitySettings, value: any) => void;
}) {
  return (
    <div className="space-y-6">
      {/* Font Size */}
      <div>
        <label className="block text-sm text-gray-300 font-medium mb-2">Font Size</label>
        <div className="grid grid-cols-4 gap-2">
          {(['small', 'medium', 'large', 'extra-large'] as const).map(size => (
            <button
              key={size}
              onClick={() => updateSetting('fontSize', size)}
              className={`
                p-2 text-center rounded border transition-colors
                ${settings.fontSize === size
                  ? 'border-blue-500 bg-blue-500/20 text-blue-400'
                  : 'border-white/10 text-gray-300 hover:border-white/20'
                }
              `}
            >
              <Type className={`w-${size === 'small' ? '3' : size === 'medium' ? '4' : size === 'large' ? '5' : '6'} h-${size === 'small' ? '3' : size === 'medium' ? '4' : size === 'large' ? '5' : '6'} mx-auto mb-1`} />
              <span className="text-xs capitalize">{size.replace('-', ' ')}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Font Family */}
      <div>
        <label className="block text-sm text-gray-300 font-medium mb-2">Font Family</label>
        <select
          value={settings.fontFamily}
          onChange={(e) => updateSetting('fontFamily', e.target.value)}
          className="w-full px-3 py-2 bg-black/20 border border-white/10 rounded text-white focus:outline-none focus:border-blue-500"
        >
          <option value="default">Default</option>
          <option value="dyslexia-friendly">Dyslexia Friendly (OpenDyslexic)</option>
          <option value="high-contrast">High Contrast (Arial Black)</option>
        </select>
      </div>

      {/* Color Scheme */}
      <div>
        <label className="block text-sm text-gray-300 font-medium mb-2">Color Scheme</label>
        <div className="grid grid-cols-2 gap-2">
          {([
            { value: 'default', label: 'Default', color: 'bg-gradient-to-r from-blue-500 to-purple-500' },
            { value: 'high-contrast', label: 'High Contrast', color: 'bg-white' },
            { value: 'dark', label: 'Dark Mode', color: 'bg-gray-900' },
            { value: 'light', label: 'Light Mode', color: 'bg-gray-100' },
            { value: 'protanopia', label: 'Protanopia', color: 'bg-gradient-to-r from-blue-500 to-yellow-500' },
            { value: 'deuteranopia', label: 'Deuteranopia', color: 'bg-gradient-to-r from-blue-500 to-orange-500' }
          ] as const).map(scheme => (
            <button
              key={scheme.value}
              onClick={() => updateSetting('colorScheme', scheme.value)}
              className={`
                p-3 rounded border transition-colors text-left
                ${settings.colorScheme === scheme.value
                  ? 'border-blue-500 bg-blue-500/20'
                  : 'border-white/10 hover:border-white/20'
                }
              `}
            >
              <div className={`w-full h-4 rounded mb-2 ${scheme.color}`} />
              <span className="text-sm text-white">{scheme.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Caption Settings */}
      <div className="space-y-4">
        <h4 className="text-white font-medium">Caption Settings</h4>
        
        <div>
          <label className="block text-sm text-gray-300 mb-2">Caption Size</label>
          <select
            value={settings.captionSize}
            onChange={(e) => updateSetting('captionSize', e.target.value)}
            className="w-full px-3 py-2 bg-black/20 border border-white/10 rounded text-white focus:outline-none focus:border-blue-500"
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
            <option value="extra-large">Extra Large</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={settings.showConfidence}
                onChange={(e) => updateSetting('showConfidence', e.target.checked)}
                className="rounded"
              />
              <span className="text-sm text-gray-300">Show Confidence</span>
            </label>
          </div>
          
          <div>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={settings.showTimestamps}
                onChange={(e) => updateSetting('showTimestamps', e.target.checked)}
                className="rounded"
              />
              <span className="text-sm text-gray-300">Show Timestamps</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}

// Audio Settings Component
function AudioSettings({ 
  settings, 
  updateSetting 
}: { 
  settings: AccessibilitySettings; 
  updateSetting: (key: keyof AccessibilitySettings, value: any) => void;
}) {
  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.audioDescriptions}
              onChange={(e) => updateSetting('audioDescriptions', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Audio Descriptions</span>
              <p className="text-sm text-gray-400">Describe visual elements for screen readers</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.soundEffects}
              onChange={(e) => updateSetting('soundEffects', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Sound Effects</span>
              <p className="text-sm text-gray-400">Play sounds for UI interactions</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.voiceAnnouncements}
              onChange={(e) => updateSetting('voiceAnnouncements', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Voice Announcements</span>
              <p className="text-sm text-gray-400">Announce important status changes</p>
            </div>
          </label>
        </div>
      </div>
    </div>
  );
}

// Motor Settings Component
function MotorSettings({ 
  settings, 
  updateSetting 
}: { 
  settings: AccessibilitySettings; 
  updateSetting: (key: keyof AccessibilitySettings, value: any) => void;
}) {
  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.reduceMotion}
              onChange={(e) => updateSetting('reduceMotion', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Reduce Motion</span>
              <p className="text-sm text-gray-400">Minimize animations and transitions</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.largerClickTargets}
              onChange={(e) => updateSetting('largerClickTargets', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Larger Click Targets</span>
              <p className="text-sm text-gray-400">Increase button and link sizes</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.stickyHover}
              onChange={(e) => updateSetting('stickyHover', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Sticky Hover</span>
              <p className="text-sm text-gray-400">Keep hover states active longer</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.keyboardNavigation}
              onChange={(e) => updateSetting('keyboardNavigation', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Enhanced Keyboard Navigation</span>
              <p className="text-sm text-gray-400">Improve keyboard accessibility</p>
            </div>
          </label>
        </div>
      </div>
    </div>
  );
}

// Cognitive Settings Component
function CognitiveSettings({ 
  settings, 
  updateSetting 
}: { 
  settings: AccessibilitySettings; 
  updateSetting: (key: keyof AccessibilitySettings, value: any) => void;
}) {
  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.simplifiedInterface}
              onChange={(e) => updateSetting('simplifiedInterface', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Simplified Interface</span>
              <p className="text-sm text-gray-400">Hide advanced features and reduce complexity</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.focusIndicators}
              onChange={(e) => updateSetting('focusIndicators', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Enhanced Focus Indicators</span>
              <p className="text-sm text-gray-400">Stronger visual focus indicators</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.autoSave}
              onChange={(e) => updateSetting('autoSave', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Auto Save</span>
              <p className="text-sm text-gray-400">Automatically save progress</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.confirmActions}
              onChange={(e) => updateSetting('confirmActions', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Confirm Actions</span>
              <p className="text-sm text-gray-400">Ask for confirmation on important actions</p>
            </div>
          </label>
        </div>

        <div>
          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={settings.screenReaderOptimized}
              onChange={(e) => updateSetting('screenReaderOptimized', e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-white font-medium">Screen Reader Optimized</span>
              <p className="text-sm text-gray-400">Optimize for screen reader users</p>
            </div>
          </label>
        </div>
      </div>
    </div>
  );
}

// Preview Component
function AccessibilityPreview({ settings }: { settings: AccessibilitySettings }) {
  return (
    <div className="space-y-4">
      <h4 className="text-white font-medium">Preview</h4>
      
      <div className="p-4 bg-black/10 rounded-lg">
        <div 
          className={`
            ${settings.fontSize === 'small' ? 'text-sm' : 
              settings.fontSize === 'medium' ? 'text-base' :
              settings.fontSize === 'large' ? 'text-lg' : 'text-xl'}
            ${settings.fontFamily === 'dyslexia-friendly' ? 'font-mono' : 
              settings.fontFamily === 'high-contrast' ? 'font-bold' : 'font-normal'}
          `}
        >
          <p className="text-white mb-2">Sample caption text with your settings applied.</p>
          
          {settings.showConfidence && (
            <span className="text-green-400 text-sm">95% confidence</span>
          )}
          
          {settings.showTimestamps && (
            <span className="text-gray-400 text-sm ml-2">[12:34:56]</span>
          )}
        </div>
        
        <div className="mt-4 flex space-x-2">
          <button 
            className={`
              px-4 py-2 rounded transition-colors
              ${settings.largerClickTargets ? 'px-6 py-3' : 'px-4 py-2'}
              ${settings.focusIndicators ? 'focus:ring-4 focus:ring-blue-500/50' : ''}
              bg-blue-500 hover:bg-blue-600 text-white
            `}
          >
            Sample Button
          </button>
        </div>
      </div>
    </div>
  );
}

// Apply accessibility settings to document
function applyAccessibilitySettings(settings: AccessibilitySettings) {
  const root = document.documentElement;
  
  // Font size
  const fontSizeMap = {
    'small': '14px',
    'medium': '16px',
    'large': '18px',
    'extra-large': '20px'
  };
  root.style.setProperty('--font-size-base', fontSizeMap[settings.fontSize]);
  
  // Font family
  if (settings.fontFamily === 'dyslexia-friendly') {
    root.style.setProperty('--font-family-base', 'OpenDyslexic, monospace');
  } else if (settings.fontFamily === 'high-contrast') {
    root.style.setProperty('--font-family-base', 'Arial Black, sans-serif');
  } else {
    root.style.setProperty('--font-family-base', 'system-ui, sans-serif');
  }
  
  // Reduce motion
  if (settings.reduceMotion) {
    root.style.setProperty('--animation-duration', '0.01ms');
    root.style.setProperty('--transition-duration', '0.01ms');
  } else {
    root.style.setProperty('--animation-duration', '');
    root.style.setProperty('--transition-duration', '');
  }
  
  // Color scheme
  root.setAttribute('data-color-scheme', settings.colorScheme);
  
  // Focus indicators
  if (settings.focusIndicators) {
    root.style.setProperty('--focus-ring-width', '3px');
    root.style.setProperty('--focus-ring-color', 'rgb(59 130 246 / 0.5)');
  }
}
