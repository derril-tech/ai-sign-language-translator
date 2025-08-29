'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  Eye, 
  EyeOff, 
  Clock, 
  Trash2, 
  Server, 
  ServerOff,
  Lock,
  Unlock,
  AlertTriangle,
  CheckCircle,
  Settings,
  Download,
  HardDrive
} from 'lucide-react';

interface PrivacySettings {
  faceBlurring: boolean;
  localOnlyMode: boolean;
  autoDeleteEnabled: boolean;
  autoDeleteHours: number;
  dataRetention: 'session' | '24h' | '7d' | '30d' | 'never';
  anonymizeTranscripts: boolean;
  encryptStorage: boolean;
  allowAnalytics: boolean;
  shareWithResearchers: boolean;
}

interface PrivacyControlsProps {
  settings: PrivacySettings;
  onSettingsChange: (settings: PrivacySettings) => void;
  isSessionActive: boolean;
  className?: string;
}

export function PrivacyControls({
  settings,
  onSettingsChange,
  isSessionActive,
  className = ''
}: PrivacyControlsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [pendingDeletion, setPendingDeletion] = useState<Date | null>(null);
  const [storageUsage, setStorageUsage] = useState({
    used: 0,
    total: 0,
    sessions: 0
  });

  // Calculate auto-delete time
  useEffect(() => {
    if (settings.autoDeleteEnabled && settings.autoDeleteHours > 0) {
      const deleteTime = new Date();
      deleteTime.setHours(deleteTime.getHours() + settings.autoDeleteHours);
      setPendingDeletion(deleteTime);
    } else {
      setPendingDeletion(null);
    }
  }, [settings.autoDeleteEnabled, settings.autoDeleteHours]);

  // Mock storage usage calculation
  useEffect(() => {
    // In production, this would query actual storage usage
    setStorageUsage({
      used: 45.2, // MB
      total: 100, // MB
      sessions: 12
    });
  }, []);

  // Handle setting changes
  const updateSetting = useCallback((key: keyof PrivacySettings, value: any) => {
    const newSettings = { ...settings, [key]: value };
    onSettingsChange(newSettings);
  }, [settings, onSettingsChange]);

  // Handle data deletion
  const handleDeleteAllData = useCallback(async () => {
    if (window.confirm('Are you sure you want to delete all stored data? This action cannot be undone.')) {
      // In production, this would call the deletion API
      console.log('Deleting all user data...');
      
      // Reset storage usage
      setStorageUsage({
        used: 0,
        total: 100,
        sessions: 0
      });
    }
  }, []);

  // Handle data export before deletion
  const handleExportAndDelete = useCallback(async () => {
    if (window.confirm('Export all data before deletion? This will download your data and then delete it from our servers.')) {
      // In production, this would trigger export then deletion
      console.log('Exporting and deleting data...');
    }
  }, []);

  // Format time remaining
  const formatTimeRemaining = (date: Date): string => {
    const now = new Date();
    const diff = date.getTime() - now.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  // Get privacy level indicator
  const getPrivacyLevel = (): { level: string; color: string; score: number } => {
    let score = 0;
    
    if (settings.faceBlurring) score += 20;
    if (settings.localOnlyMode) score += 25;
    if (settings.autoDeleteEnabled) score += 15;
    if (settings.anonymizeTranscripts) score += 20;
    if (settings.encryptStorage) score += 15;
    if (!settings.allowAnalytics) score += 5;
    
    if (score >= 80) return { level: 'Maximum', color: 'text-green-400', score };
    if (score >= 60) return { level: 'High', color: 'text-blue-400', score };
    if (score >= 40) return { level: 'Medium', color: 'text-yellow-400', score };
    return { level: 'Basic', color: 'text-red-400', score };
  };

  const privacyLevel = getPrivacyLevel();

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <Shield className="w-5 h-5 text-blue-400" />
            <h3 className="text-white font-medium">Privacy Controls</h3>
            <div className={`px-2 py-1 rounded text-xs ${privacyLevel.color} bg-current bg-opacity-10`}>
              {privacyLevel.level} ({privacyLevel.score}%)
            </div>
          </div>

          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Advanced settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>

        {/* Privacy Level Indicator */}
        <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${privacyLevel.score}%` }}
            className={`h-2 rounded-full ${
              privacyLevel.score >= 80 ? 'bg-green-500' :
              privacyLevel.score >= 60 ? 'bg-blue-500' :
              privacyLevel.score >= 40 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
          />
        </div>
      </div>

      {/* Core Privacy Settings */}
      <div className="p-4 space-y-4">
        
        {/* Face Blurring */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {settings.faceBlurring ? (
              <EyeOff className="w-4 h-4 text-green-400" />
            ) : (
              <Eye className="w-4 h-4 text-gray-400" />
            )}
            <div>
              <span className="text-white font-medium">Face Blurring</span>
              <p className="text-xs text-gray-400">Automatically blur faces in video feed</p>
            </div>
          </div>
          
          <button
            onClick={() => updateSetting('faceBlurring', !settings.faceBlurring)}
            disabled={isSessionActive}
            className={`
              relative inline-flex h-6 w-11 items-center rounded-full transition-colors
              ${settings.faceBlurring ? 'bg-green-600' : 'bg-gray-600'}
              ${isSessionActive ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            <span
              className={`
                inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                ${settings.faceBlurring ? 'translate-x-6' : 'translate-x-1'}
              `}
            />
          </button>
        </div>

        {/* Local-Only Mode */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {settings.localOnlyMode ? (
              <HardDrive className="w-4 h-4 text-green-400" />
            ) : (
              <Server className="w-4 h-4 text-gray-400" />
            )}
            <div>
              <span className="text-white font-medium">Local-Only Mode</span>
              <p className="text-xs text-gray-400">Process everything locally, no cloud sync</p>
            </div>
          </div>
          
          <button
            onClick={() => updateSetting('localOnlyMode', !settings.localOnlyMode)}
            className={`
              relative inline-flex h-6 w-11 items-center rounded-full transition-colors
              ${settings.localOnlyMode ? 'bg-green-600' : 'bg-gray-600'}
            `}
          >
            <span
              className={`
                inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                ${settings.localOnlyMode ? 'translate-x-6' : 'translate-x-1'}
              `}
            />
          </button>
        </div>

        {/* Auto-Delete */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Clock className="w-4 h-4 text-blue-400" />
              <div>
                <span className="text-white font-medium">Auto-Delete</span>
                <p className="text-xs text-gray-400">Automatically delete data after specified time</p>
              </div>
            </div>
            
            <button
              onClick={() => updateSetting('autoDeleteEnabled', !settings.autoDeleteEnabled)}
              className={`
                relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                ${settings.autoDeleteEnabled ? 'bg-blue-600' : 'bg-gray-600'}
              `}
            >
              <span
                className={`
                  inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                  ${settings.autoDeleteEnabled ? 'translate-x-6' : 'translate-x-1'}
                `}
              />
            </button>
          </div>

          {settings.autoDeleteEnabled && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="ml-7 space-y-2"
            >
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-300">Delete after:</span>
                <select
                  value={settings.autoDeleteHours}
                  onChange={(e) => updateSetting('autoDeleteHours', parseInt(e.target.value))}
                  className="px-2 py-1 bg-black/20 border border-white/10 rounded text-white text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value={1}>1 hour</option>
                  <option value={6}>6 hours</option>
                  <option value={24}>24 hours</option>
                  <option value={168}>7 days</option>
                  <option value={720}>30 days</option>
                </select>
              </div>

              {pendingDeletion && (
                <div className="text-xs text-yellow-400 flex items-center space-x-1">
                  <AlertTriangle className="w-3 h-3" />
                  <span>Data will be deleted in {formatTimeRemaining(pendingDeletion)}</span>
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Storage Usage */}
        <div className="p-3 bg-gray-500/10 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">Storage Usage</span>
            <span className="text-sm text-white">
              {storageUsage.used.toFixed(1)} MB / {storageUsage.total} MB
            </span>
          </div>
          
          <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
            <div
              className={`h-2 rounded-full ${
                storageUsage.used / storageUsage.total > 0.8 ? 'bg-red-500' :
                storageUsage.used / storageUsage.total > 0.6 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
              style={{ width: `${(storageUsage.used / storageUsage.total) * 100}%` }}
            />
          </div>
          
          <div className="flex items-center justify-between text-xs text-gray-400">
            <span>{storageUsage.sessions} sessions stored</span>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleExportAndDelete}
                className="text-blue-400 hover:text-blue-300 transition-colors"
              >
                Export & Delete
              </button>
              <button
                onClick={handleDeleteAllData}
                className="text-red-400 hover:text-red-300 transition-colors"
              >
                Delete All
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Advanced Settings */}
      <AnimatePresence>
        {showAdvanced && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-white/10 p-4 space-y-4"
          >
            <h4 className="text-white font-medium mb-3">Advanced Privacy Settings</h4>

            {/* Data Retention */}
            <div className="space-y-2">
              <label className="text-sm text-gray-300">Data Retention Policy</label>
              <select
                value={settings.dataRetention}
                onChange={(e) => updateSetting('dataRetention', e.target.value)}
                className="w-full px-3 py-2 bg-black/20 border border-white/10 rounded text-white focus:outline-none focus:border-blue-500"
              >
                <option value="session">Session only (delete when closed)</option>
                <option value="24h">24 hours</option>
                <option value="7d">7 days</option>
                <option value="30d">30 days</option>
                <option value="never">Never delete (manual only)</option>
              </select>
            </div>

            {/* Anonymize Transcripts */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-white font-medium">Anonymize Transcripts</span>
                <p className="text-xs text-gray-400">Remove identifying information from transcripts</p>
              </div>
              
              <button
                onClick={() => updateSetting('anonymizeTranscripts', !settings.anonymizeTranscripts)}
                className={`
                  relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                  ${settings.anonymizeTranscripts ? 'bg-green-600' : 'bg-gray-600'}
                `}
              >
                <span
                  className={`
                    inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                    ${settings.anonymizeTranscripts ? 'translate-x-6' : 'translate-x-1'}
                  `}
                />
              </button>
            </div>

            {/* Encrypt Storage */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                {settings.encryptStorage ? (
                  <Lock className="w-4 h-4 text-green-400" />
                ) : (
                  <Unlock className="w-4 h-4 text-red-400" />
                )}
                <div>
                  <span className="text-white font-medium">Encrypt Local Storage</span>
                  <p className="text-xs text-gray-400">Encrypt data stored on this device</p>
                </div>
              </div>
              
              <button
                onClick={() => updateSetting('encryptStorage', !settings.encryptStorage)}
                className={`
                  relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                  ${settings.encryptStorage ? 'bg-green-600' : 'bg-gray-600'}
                `}
              >
                <span
                  className={`
                    inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                    ${settings.encryptStorage ? 'translate-x-6' : 'translate-x-1'}
                  `}
                />
              </button>
            </div>

            {/* Analytics */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-white font-medium">Allow Analytics</span>
                <p className="text-xs text-gray-400">Share anonymous usage data to improve the service</p>
              </div>
              
              <button
                onClick={() => updateSetting('allowAnalytics', !settings.allowAnalytics)}
                className={`
                  relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                  ${settings.allowAnalytics ? 'bg-blue-600' : 'bg-gray-600'}
                `}
              >
                <span
                  className={`
                    inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                    ${settings.allowAnalytics ? 'translate-x-6' : 'translate-x-1'}
                  `}
                />
              </button>
            </div>

            {/* Research Sharing */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-white font-medium">Share with Researchers</span>
                <p className="text-xs text-gray-400">Contribute anonymized data to ASL research</p>
              </div>
              
              <button
                onClick={() => updateSetting('shareWithResearchers', !settings.shareWithResearchers)}
                className={`
                  relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                  ${settings.shareWithResearchers ? 'bg-purple-600' : 'bg-gray-600'}
                `}
              >
                <span
                  className={`
                    inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                    ${settings.shareWithResearchers ? 'translate-x-6' : 'translate-x-1'}
                  `}
                />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Privacy Warnings */}
      <div className="p-4 border-t border-white/10">
        {settings.localOnlyMode && (
          <div className="mb-2 p-2 bg-green-500/10 border border-green-500/20 rounded text-green-400 text-xs flex items-center space-x-2">
            <CheckCircle className="w-3 h-3" />
            <span>Local-only mode active. Your data stays on this device.</span>
          </div>
        )}

        {!settings.faceBlurring && !settings.localOnlyMode && (
          <div className="mb-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-yellow-400 text-xs flex items-center space-x-2">
            <AlertTriangle className="w-3 h-3" />
            <span>Consider enabling face blurring and local-only mode for maximum privacy.</span>
          </div>
        )}

        {storageUsage.used / storageUsage.total > 0.8 && (
          <div className="p-2 bg-red-500/10 border border-red-500/20 rounded text-red-400 text-xs flex items-center space-x-2">
            <AlertTriangle className="w-3 h-3" />
            <span>Storage almost full. Consider deleting old sessions or enabling auto-delete.</span>
          </div>
        )}
      </div>
    </div>
  );
}
