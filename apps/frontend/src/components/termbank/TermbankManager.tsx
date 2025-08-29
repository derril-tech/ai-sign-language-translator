'use client';

import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Search, 
  Plus, 
  Edit, 
  Trash2, 
  Download, 
  BookOpen,
  Filter,
  Tag,
  FileText,
  AlertCircle
} from 'lucide-react';

interface TermEntry {
  id: string;
  term: string;
  definition: string;
  domain: string;
  context: string;
  confidence: number;
  usage_count: number;
  created_at: string;
  updated_at: string;
}

interface Domain {
  id: string;
  name: string;
  description: string;
  term_count: number;
  color: string;
}

interface TermbankManagerProps {
  className?: string;
}

export function TermbankManager({ className = '' }: TermbankManagerProps) {
  const [terms, setTerms] = useState<TermEntry[]>([]);
  const [domains, setDomains] = useState<Domain[]>([
    { id: 'medical', name: 'Medical', description: 'Healthcare terminology', term_count: 0, color: 'bg-red-500' },
    { id: 'legal', name: 'Legal', description: 'Legal and court terminology', term_count: 0, color: 'bg-blue-500' },
    { id: 'education', name: 'Education', description: 'Academic and educational terms', term_count: 0, color: 'bg-green-500' },
    { id: 'business', name: 'Business', description: 'Corporate and business terms', term_count: 0, color: 'bg-purple-500' },
    { id: 'general', name: 'General', description: 'General vocabulary', term_count: 0, color: 'bg-gray-500' }
  ]);
  
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<string>('all');
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingTerm, setEditingTerm] = useState<TermEntry | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Filter terms based on search and domain
  const filteredTerms = terms.filter(term => {
    const matchesSearch = term.term.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         term.definition.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesDomain = selectedDomain === 'all' || term.domain === selectedDomain;
    return matchesSearch && matchesDomain;
  });

  // Add new term
  const handleAddTerm = useCallback((termData: Omit<TermEntry, 'id' | 'created_at' | 'updated_at' | 'usage_count'>) => {
    const newTerm: TermEntry = {
      ...termData,
      id: `term_${Date.now()}`,
      usage_count: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    
    setTerms(prev => [...prev, newTerm]);
    setShowAddForm(false);
    
    // Update domain count
    setDomains(prev => prev.map(domain => 
      domain.id === newTerm.domain 
        ? { ...domain, term_count: domain.term_count + 1 }
        : domain
    ));
  }, []);

  // Edit term
  const handleEditTerm = useCallback((updatedTerm: TermEntry) => {
    setTerms(prev => prev.map(term => 
      term.id === updatedTerm.id 
        ? { ...updatedTerm, updated_at: new Date().toISOString() }
        : term
    ));
    setEditingTerm(null);
  }, []);

  // Delete term
  const handleDeleteTerm = useCallback((termId: string) => {
    const term = terms.find(t => t.id === termId);
    if (term) {
      setTerms(prev => prev.filter(t => t.id !== termId));
      
      // Update domain count
      setDomains(prev => prev.map(domain => 
        domain.id === term.domain 
          ? { ...domain, term_count: Math.max(0, domain.term_count - 1) }
          : domain
      ));
    }
  }, [terms]);

  // Handle file upload
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    
    try {
      const text = await file.text();
      let importedTerms: TermEntry[] = [];
      
      if (file.name.endsWith('.json')) {
        // JSON format
        const data = JSON.parse(text);
        importedTerms = Array.isArray(data) ? data : [data];
      } else if (file.name.endsWith('.csv')) {
        // CSV format
        const lines = text.split('\n');
        const headers = lines[0].split(',');
        
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',');
          if (values.length >= 3) {
            importedTerms.push({
              id: `imported_${Date.now()}_${i}`,
              term: values[0]?.trim() || '',
              definition: values[1]?.trim() || '',
              domain: values[2]?.trim() || 'general',
              context: values[3]?.trim() || '',
              confidence: parseFloat(values[4]) || 0.8,
              usage_count: 0,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            });
          }
        }
      }
      
      // Add imported terms
      setTerms(prev => [...prev, ...importedTerms]);
      
      // Update domain counts
      const domainCounts = new Map<string, number>();
      importedTerms.forEach(term => {
        domainCounts.set(term.domain, (domainCounts.get(term.domain) || 0) + 1);
      });
      
      setDomains(prev => prev.map(domain => ({
        ...domain,
        term_count: domain.term_count + (domainCounts.get(domain.id) || 0)
      })));
      
    } catch (error) {
      console.error('Error importing terms:', error);
      alert('Error importing file. Please check the format.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, []);

  // Export terms
  const handleExport = useCallback((format: 'json' | 'csv') => {
    let content: string;
    let filename: string;
    let mimeType: string;
    
    if (format === 'json') {
      content = JSON.stringify(filteredTerms, null, 2);
      filename = `termbank_${selectedDomain}_${new Date().toISOString().split('T')[0]}.json`;
      mimeType = 'application/json';
    } else {
      const headers = ['term', 'definition', 'domain', 'context', 'confidence'];
      const rows = filteredTerms.map(term => [
        term.term,
        term.definition,
        term.domain,
        term.context,
        term.confidence.toString()
      ]);
      
      content = [headers, ...rows].map(row => row.join(',')).join('\n');
      filename = `termbank_${selectedDomain}_${new Date().toISOString().split('T')[0]}.csv`;
      mimeType = 'text/csv';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [filteredTerms, selectedDomain]);

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <BookOpen className="w-6 h-6 text-blue-400" />
            <h2 className="text-xl font-bold text-white">Terminology Manager</h2>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleExport('json')}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Export as JSON"
            >
              <Download className="w-4 h-4" />
            </button>
            
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="flex items-center space-x-2 px-3 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors disabled:opacity-50"
            >
              <Upload className="w-4 h-4" />
              <span className="text-sm">Import</span>
            </button>
            
            <button
              onClick={() => setShowAddForm(true)}
              className="flex items-center space-x-2 px-3 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span className="text-sm">Add Term</span>
            </button>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search terms or definitions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-black/20 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <select
            value={selectedDomain}
            onChange={(e) => setSelectedDomain(e.target.value)}
            className="px-3 py-2 bg-black/20 border border-white/10 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="all">All Domains</option>
            {domains.map(domain => (
              <option key={domain.id} value={domain.id}>
                {domain.name} ({domain.term_count})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Domain Overview */}
      <div className="p-6 border-b border-white/10">
        <h3 className="text-lg font-medium text-white mb-3">Domain Overview</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {domains.map(domain => (
            <motion.button
              key={domain.id}
              onClick={() => setSelectedDomain(domain.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`p-3 rounded-lg border transition-all ${
                selectedDomain === domain.id
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-white/10 bg-black/20 hover:bg-white/5'
              }`}
            >
              <div className="flex items-center space-x-2 mb-2">
                <div className={`w-3 h-3 rounded-full ${domain.color}`} />
                <span className="text-white font-medium text-sm">{domain.name}</span>
              </div>
              <div className="text-2xl font-bold text-white">{domain.term_count}</div>
              <div className="text-xs text-gray-400">{domain.description}</div>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Terms List */}
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-white">
            Terms ({filteredTerms.length})
          </h3>
          
          {filteredTerms.length > 0 && (
            <div className="flex items-center space-x-2">
              <button
                onClick={() => handleExport('csv')}
                className="text-xs text-gray-400 hover:text-white transition-colors"
              >
                Export CSV
              </button>
            </div>
          )}
        </div>

        {/* Terms Grid */}
        <div className="space-y-3 max-h-96 overflow-y-auto">
          <AnimatePresence>
            {filteredTerms.map(term => (
              <motion.div
                key={term.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="bg-black/20 border border-white/10 rounded-lg p-4"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h4 className="text-white font-medium">{term.term}</h4>
                      <span className={`px-2 py-1 text-xs rounded ${
                        domains.find(d => d.id === term.domain)?.color || 'bg-gray-500'
                      } text-white`}>
                        {domains.find(d => d.id === term.domain)?.name || term.domain}
                      </span>
                      <span className="text-xs text-gray-400">
                        {Math.round(term.confidence * 100)}% confidence
                      </span>
                    </div>
                    
                    <p className="text-gray-300 text-sm mb-2">{term.definition}</p>
                    
                    {term.context && (
                      <p className="text-gray-400 text-xs italic">Context: {term.context}</p>
                    )}
                    
                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                      <span>Used {term.usage_count} times</span>
                      <span>Created {new Date(term.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setEditingTerm(term)}
                      className="p-2 text-gray-400 hover:text-white transition-colors"
                    >
                      <Edit className="w-4 h-4" />
                    </button>
                    
                    <button
                      onClick={() => handleDeleteTerm(term.id)}
                      className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {filteredTerms.length === 0 && (
            <div className="text-center py-8">
              <FileText className="w-12 h-12 text-gray-500 mx-auto mb-3" />
              <p className="text-gray-400">
                {searchQuery || selectedDomain !== 'all' 
                  ? 'No terms match your search criteria'
                  : 'No terms added yet. Import a termbank or add terms manually.'
                }
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,.csv"
        onChange={handleFileUpload}
        className="hidden"
      />

      {/* Add/Edit Term Modal */}
      <TermFormModal
        isOpen={showAddForm || editingTerm !== null}
        term={editingTerm}
        domains={domains}
        onSave={editingTerm ? handleEditTerm : handleAddTerm}
        onClose={() => {
          setShowAddForm(false);
          setEditingTerm(null);
        }}
      />
    </div>
  );
}

// Term Form Modal Component
interface TermFormModalProps {
  isOpen: boolean;
  term: TermEntry | null;
  domains: Domain[];
  onSave: (term: any) => void;
  onClose: () => void;
}

function TermFormModal({ isOpen, term, domains, onSave, onClose }: TermFormModalProps) {
  const [formData, setFormData] = useState({
    term: '',
    definition: '',
    domain: 'general',
    context: '',
    confidence: 0.8
  });

  // Reset form when modal opens/closes
  React.useEffect(() => {
    if (isOpen) {
      if (term) {
        setFormData({
          term: term.term,
          definition: term.definition,
          domain: term.domain,
          context: term.context,
          confidence: term.confidence
        });
      } else {
        setFormData({
          term: '',
          definition: '',
          domain: 'general',
          context: '',
          confidence: 0.8
        });
      }
    }
  }, [isOpen, term]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (term) {
      onSave({ ...term, ...formData });
    } else {
      onSave(formData);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-gray-800 rounded-lg border border-white/10 p-6 w-full max-w-md mx-4"
      >
        <h3 className="text-lg font-medium text-white mb-4">
          {term ? 'Edit Term' : 'Add New Term'}
        </h3>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-gray-300 mb-1">Term</label>
            <input
              type="text"
              value={formData.term}
              onChange={(e) => setFormData(prev => ({ ...prev, term: e.target.value }))}
              className="w-full px-3 py-2 bg-black/20 border border-white/10 rounded text-white focus:outline-none focus:border-blue-500"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-1">Definition</label>
            <textarea
              value={formData.definition}
              onChange={(e) => setFormData(prev => ({ ...prev, definition: e.target.value }))}
              className="w-full px-3 py-2 bg-black/20 border border-white/10 rounded text-white focus:outline-none focus:border-blue-500 h-20"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-1">Domain</label>
            <select
              value={formData.domain}
              onChange={(e) => setFormData(prev => ({ ...prev, domain: e.target.value }))}
              className="w-full px-3 py-2 bg-black/20 border border-white/10 rounded text-white focus:outline-none focus:border-blue-500"
            >
              {domains.map(domain => (
                <option key={domain.id} value={domain.id}>
                  {domain.name}
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-1">Context (Optional)</label>
            <input
              type="text"
              value={formData.context}
              onChange={(e) => setFormData(prev => ({ ...prev, context: e.target.value }))}
              className="w-full px-3 py-2 bg-black/20 border border-white/10 rounded text-white focus:outline-none focus:border-blue-500"
              placeholder="Usage context or example"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-1">
              Confidence: {Math.round(formData.confidence * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={formData.confidence}
              onChange={(e) => setFormData(prev => ({ ...prev, confidence: parseFloat(e.target.value) }))}
              className="w-full accent-blue-500"
            />
          </div>
          
          <div className="flex items-center justify-end space-x-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            
            <button
              type="submit"
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors"
            >
              {term ? 'Update' : 'Add'} Term
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
}
