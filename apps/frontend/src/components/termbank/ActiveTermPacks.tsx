'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Package, 
  Plus, 
  Minus, 
  Settings, 
  Download, 
  Upload,
  Search,
  Filter,
  Star,
  Clock,
  TrendingUp,
  AlertCircle
} from 'lucide-react';

interface TermPack {
  id: string;
  name: string;
  description: string;
  domain: string;
  version: string;
  termCount: number;
  isActive: boolean;
  priority: number; // Higher priority packs are checked first
  lastUsed: number;
  usageCount: number;
  confidence: number;
  author: string;
  tags: string[];
  terms: TermPackEntry[];
}

interface TermPackEntry {
  term: string;
  definition: string;
  context: string;
  confidence: number;
  alternatives: string[];
  usage_examples: string[];
}

interface ActiveTermPacksProps {
  onTermMatch: (term: string, pack: TermPack, entry: TermPackEntry) => void;
  onPackActivation: (packId: string, active: boolean) => void;
  className?: string;
}

export function ActiveTermPacks({
  onTermMatch,
  onPackActivation,
  className = ''
}: ActiveTermPacksProps) {
  const [termPacks, setTermPacks] = useState<TermPack[]>([]);
  const [activePacks, setActivePacks] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<string>('all');
  const [showPackManager, setShowPackManager] = useState(false);
  const [recentMatches, setRecentMatches] = useState<Array<{
    term: string;
    pack: string;
    timestamp: number;
    confidence: number;
  }>>([]);

  // Mock term packs data
  useEffect(() => {
    const mockPacks: TermPack[] = [
      {
        id: 'medical_basic',
        name: 'Medical Basics',
        description: 'Essential medical terminology for healthcare settings',
        domain: 'medical',
        version: '2.1.0',
        termCount: 245,
        isActive: true,
        priority: 9,
        lastUsed: Date.now() - 3600000,
        usageCount: 156,
        confidence: 0.92,
        author: 'Medical ASL Consortium',
        tags: ['healthcare', 'basic', 'certified'],
        terms: [
          {
            term: 'DOCTOR',
            definition: 'Medical professional who diagnoses and treats patients',
            context: 'Healthcare setting, formal medical consultation',
            confidence: 0.95,
            alternatives: ['PHYSICIAN', 'MD'],
            usage_examples: ['I need to see a DOCTOR', 'The DOCTOR will see you now']
          }
        ]
      },
      {
        id: 'legal_court',
        name: 'Legal & Court',
        description: 'Legal terminology for court proceedings and legal consultations',
        domain: 'legal',
        version: '1.8.2',
        termCount: 189,
        isActive: false,
        priority: 7,
        lastUsed: Date.now() - 86400000,
        usageCount: 89,
        confidence: 0.88,
        author: 'Legal ASL Foundation',
        tags: ['legal', 'court', 'certified'],
        terms: []
      },
      {
        id: 'education_k12',
        name: 'K-12 Education',
        description: 'Educational terminology for elementary and secondary schools',
        domain: 'education',
        version: '3.0.1',
        termCount: 312,
        isActive: true,
        priority: 8,
        lastUsed: Date.now() - 7200000,
        usageCount: 203,
        confidence: 0.90,
        author: 'Educational ASL Network',
        tags: ['education', 'k12', 'school'],
        terms: []
      },
      {
        id: 'business_corporate',
        name: 'Corporate Business',
        description: 'Business terminology for corporate environments',
        domain: 'business',
        version: '1.5.0',
        termCount: 167,
        isActive: false,
        priority: 6,
        lastUsed: Date.now() - 172800000,
        usageCount: 45,
        confidence: 0.85,
        author: 'Business ASL Alliance',
        tags: ['business', 'corporate', 'professional'],
        terms: []
      }
    ];

    setTermPacks(mockPacks);
    setActivePacks(new Set(mockPacks.filter(p => p.isActive).map(p => p.id)));
  }, []);

  // Filter packs based on search and domain
  const filteredPacks = termPacks.filter(pack => {
    const matchesSearch = pack.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         pack.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         pack.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesDomain = selectedDomain === 'all' || pack.domain === selectedDomain;
    return matchesSearch && matchesDomain;
  });

  // Toggle pack activation
  const togglePackActivation = useCallback((packId: string) => {
    setActivePacks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(packId)) {
        newSet.delete(packId);
      } else {
        newSet.add(packId);
      }
      return newSet;
    });

    setTermPacks(prev => prev.map(pack => 
      pack.id === packId 
        ? { ...pack, isActive: !pack.isActive, lastUsed: Date.now() }
        : pack
    ));

    onPackActivation(packId, !activePacks.has(packId));
  }, [activePacks, onPackActivation]);

  // Search for term matches across active packs
  const searchTerms = useCallback((query: string): Array<{
    term: string;
    pack: TermPack;
    entry: TermPackEntry;
    score: number;
  }> => {
    const results: Array<{
      term: string;
      pack: TermPack;
      entry: TermPackEntry;
      score: number;
    }> = [];

    // Get active packs sorted by priority
    const activePacksList = termPacks
      .filter(pack => activePacks.has(pack.id))
      .sort((a, b) => b.priority - a.priority);

    activePacksList.forEach(pack => {
      pack.terms.forEach(entry => {
        // Calculate match score
        let score = 0;
        const queryLower = query.toLowerCase();
        const termLower = entry.term.toLowerCase();

        // Exact match
        if (termLower === queryLower) {
          score = 1.0;
        }
        // Starts with
        else if (termLower.startsWith(queryLower)) {
          score = 0.9;
        }
        // Contains
        else if (termLower.includes(queryLower)) {
          score = 0.7;
        }
        // Alternative matches
        else if (entry.alternatives.some(alt => alt.toLowerCase().includes(queryLower))) {
          score = 0.6;
        }
        // Definition matches
        else if (entry.definition.toLowerCase().includes(queryLower)) {
          score = 0.4;
        }

        // Apply pack confidence and priority weighting
        score *= entry.confidence * (pack.priority / 10) * pack.confidence;

        if (score > 0.3) {
          results.push({ term: entry.term, pack, entry, score });
        }
      });
    });

    return results.sort((a, b) => b.score - a.score);
  }, [termPacks, activePacks]);

  // Handle term lookup
  const handleTermLookup = useCallback((query: string) => {
    const matches = searchTerms(query);
    if (matches.length > 0) {
      const bestMatch = matches[0];
      onTermMatch(bestMatch.term, bestMatch.pack, bestMatch.entry);
      
      // Add to recent matches
      setRecentMatches(prev => [
        {
          term: bestMatch.term,
          pack: bestMatch.pack.name,
          timestamp: Date.now(),
          confidence: bestMatch.score
        },
        ...prev.slice(0, 9) // Keep last 10 matches
      ]);

      // Update pack usage
      setTermPacks(prev => prev.map(pack => 
        pack.id === bestMatch.pack.id
          ? { ...pack, usageCount: pack.usageCount + 1, lastUsed: Date.now() }
          : pack
      ));
    }
  }, [searchTerms, onTermMatch]);

  // Get domain list
  const domains = Array.from(new Set(termPacks.map(pack => pack.domain)));

  // Format last used time
  const formatLastUsed = (timestamp: number): string => {
    const diff = Date.now() - timestamp;
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    return 'Just now';
  };

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <Package className="w-5 h-5 text-green-400" />
            <h3 className="text-white font-medium">Active Term Packs</h3>
            <span className="text-sm text-gray-400">
              ({activePacks.size} active)
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowPackManager(!showPackManager)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Manage packs"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Quick Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search terms across active packs..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              if (e.target.value.length > 2) {
                handleTermLookup(e.target.value);
              }
            }}
            className="w-full pl-10 pr-4 py-2 bg-black/20 border border-white/10 rounded text-white placeholder-gray-400 focus:outline-none focus:border-green-500"
          />
        </div>
      </div>

      {/* Active Packs Summary */}
      <div className="p-4 border-b border-white/10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="text-center">
            <div className="text-lg font-bold text-green-400">
              {termPacks.filter(p => activePacks.has(p.id)).reduce((sum, p) => sum + p.termCount, 0)}
            </div>
            <div className="text-xs text-gray-400">Total Terms</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-blue-400">
              {Math.round(termPacks.filter(p => activePacks.has(p.id)).reduce((sum, p) => sum + p.confidence, 0) / Math.max(activePacks.size, 1) * 100)}%
            </div>
            <div className="text-xs text-gray-400">Avg Confidence</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-purple-400">
              {termPacks.filter(p => activePacks.has(p.id)).reduce((sum, p) => sum + p.usageCount, 0)}
            </div>
            <div className="text-xs text-gray-400">Total Usage</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-yellow-400">
              {domains.filter(domain => 
                termPacks.some(p => p.domain === domain && activePacks.has(p.id))
              ).length}
            </div>
            <div className="text-xs text-gray-400">Domains</div>
          </div>
        </div>
      </div>

      {/* Recent Matches */}
      {recentMatches.length > 0 && (
        <div className="p-4 border-b border-white/10">
          <h4 className="text-sm text-gray-300 font-medium mb-2">Recent Matches</h4>
          <div className="space-y-1 max-h-24 overflow-y-auto">
            {recentMatches.slice(0, 5).map((match, index) => (
              <div key={index} className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <span className="text-white font-medium">{match.term}</span>
                  <span className="text-gray-400">from {match.pack}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-green-400">{Math.round(match.confidence * 100)}%</span>
                  <span className="text-gray-500">{formatLastUsed(match.timestamp)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pack Manager */}
      <AnimatePresence>
        {showPackManager && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-b border-white/10"
          >
            <div className="p-4">
              <div className="flex items-center space-x-4 mb-4">
                <select
                  value={selectedDomain}
                  onChange={(e) => setSelectedDomain(e.target.value)}
                  className="px-3 py-1 bg-black/20 border border-white/10 rounded text-white text-sm focus:outline-none focus:border-green-500"
                >
                  <option value="all">All Domains</option>
                  {domains.map(domain => (
                    <option key={domain} value={domain}>
                      {domain.charAt(0).toUpperCase() + domain.slice(1)}
                    </option>
                  ))}
                </select>

                <div className="flex items-center space-x-2 text-sm text-gray-400">
                  <span>Sort by:</span>
                  <button className="text-white hover:text-green-400 transition-colors">
                    Priority
                  </button>
                </div>
              </div>

              <div className="space-y-2 max-h-48 overflow-y-auto">
                {filteredPacks.map(pack => (
                  <div
                    key={pack.id}
                    className={`p-3 rounded border transition-all ${
                      activePacks.has(pack.id)
                        ? 'border-green-500/50 bg-green-500/10'
                        : 'border-white/10 bg-black/20'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-1">
                          <h5 className="text-white font-medium">{pack.name}</h5>
                          <span className="text-xs text-gray-400">v{pack.version}</span>
                          <div className="flex items-center space-x-1">
                            {pack.tags.includes('certified') && (
                              <Star className="w-3 h-3 text-yellow-400" />
                            )}
                            <span className="text-xs text-green-400">
                              {Math.round(pack.confidence * 100)}%
                            </span>
                          </div>
                        </div>
                        
                        <p className="text-sm text-gray-300 mb-2">{pack.description}</p>
                        
                        <div className="flex items-center space-x-4 text-xs text-gray-400">
                          <span>{pack.termCount} terms</span>
                          <span>Priority: {pack.priority}</span>
                          <span>Used: {formatLastUsed(pack.lastUsed)}</span>
                          <span>{pack.usageCount} times</span>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => togglePackActivation(pack.id)}
                          className={`p-2 rounded transition-colors ${
                            activePacks.has(pack.id)
                              ? 'text-red-400 hover:text-red-300'
                              : 'text-green-400 hover:text-green-300'
                          }`}
                          title={activePacks.has(pack.id) ? 'Deactivate pack' : 'Activate pack'}
                        >
                          {activePacks.has(pack.id) ? (
                            <Minus className="w-4 h-4" />
                          ) : (
                            <Plus className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Actions */}
      <div className="p-4">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-400">
            {activePacks.size > 0 
              ? `${activePacks.size} pack(s) active and monitoring`
              : 'No active packs - activate packs to enable terminology assistance'
            }
          </div>

          <div className="flex items-center space-x-2">
            <button
              className="p-1 text-gray-400 hover:text-white transition-colors text-xs"
              title="Import pack"
            >
              <Upload className="w-3 h-3" />
            </button>
            <button
              className="p-1 text-gray-400 hover:text-white transition-colors text-xs"
              title="Export active packs"
            >
              <Download className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Performance Warning */}
        {activePacks.size > 5 && (
          <div className="mt-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-yellow-400 text-xs flex items-center space-x-2">
            <AlertCircle className="w-3 h-3" />
            <span>Many active packs may impact performance. Consider deactivating unused packs.</span>
          </div>
        )}
      </div>
    </div>
  );
}
