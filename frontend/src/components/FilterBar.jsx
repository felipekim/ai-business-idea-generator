import { useState } from 'react'
import { Search, Filter, SortAsc, SortDesc } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Badge } from '@/components/ui/badge.jsx'

export default function FilterBar({ 
  onSearch, 
  onNicheFilter, 
  onScoreFilter, 
  onSort,
  niches = [],
  currentFilters = {}
}) {
  const [searchTerm, setSearchTerm] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  
  const handleSearch = (value) => {
    setSearchTerm(value)
    onSearch(value)
  }
  
  const clearFilters = () => {
    setSearchTerm('')
    onSearch('')
    onNicheFilter('')
    onScoreFilter('')
    onSort('created_at', 'desc')
  }
  
  const activeFiltersCount = Object.values(currentFilters).filter(Boolean).length
  
  return (
    <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div className="container mx-auto px-4 py-4">
        {/* Main Filter Bar */}
        <div className="flex flex-col md:flex-row gap-4 items-center">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <Input
              placeholder="Search business ideas..."
              value={searchTerm}
              onChange={(e) => handleSearch(e.target.value)}
              className="pl-10"
            />
          </div>
          
          {/* Quick Filters */}
          <div className="flex items-center space-x-2">
            <Button
              variant={showFilters ? "default" : "outline"}
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
              className="relative"
            >
              <Filter className="h-4 w-4 mr-1" />
              Filters
              {activeFiltersCount > 0 && (
                <Badge 
                  variant="destructive" 
                  className="absolute -top-2 -right-2 h-5 w-5 p-0 flex items-center justify-center text-xs"
                >
                  {activeFiltersCount}
                </Badge>
              )}
            </Button>
            
            <Select onValueChange={(value) => {
              const [sortBy, order] = value.split('-')
              onSort(sortBy, order)
            }}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="created_at-desc">
                  <div className="flex items-center">
                    <SortDesc className="h-4 w-4 mr-2" />
                    Newest First
                  </div>
                </SelectItem>
                <SelectItem value="created_at-asc">
                  <div className="flex items-center">
                    <SortAsc className="h-4 w-4 mr-2" />
                    Oldest First
                  </div>
                </SelectItem>
                <SelectItem value="score-desc">
                  <div className="flex items-center">
                    <SortDesc className="h-4 w-4 mr-2" />
                    Highest Score
                  </div>
                </SelectItem>
                <SelectItem value="score-asc">
                  <div className="flex items-center">
                    <SortAsc className="h-4 w-4 mr-2" />
                    Lowest Score
                  </div>
                </SelectItem>
                <SelectItem value="cost-asc">
                  <div className="flex items-center">
                    <SortAsc className="h-4 w-4 mr-2" />
                    Lowest Cost
                  </div>
                </SelectItem>
                <SelectItem value="cost-desc">
                  <div className="flex items-center">
                    <SortDesc className="h-4 w-4 mr-2" />
                    Highest Cost
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        
        {/* Expanded Filters */}
        {showFilters && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Niche Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Niche
                </label>
                <Select onValueChange={onNicheFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="All niches" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All niches</SelectItem>
                    {niches.map((niche) => (
                      <SelectItem key={niche} value={niche}>
                        {niche}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              {/* Score Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Minimum Score
                </label>
                <Select onValueChange={onScoreFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="Any score" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">Any score</SelectItem>
                    <SelectItem value="8">8+ (Excellent)</SelectItem>
                    <SelectItem value="7">7+ (Very Good)</SelectItem>
                    <SelectItem value="6">6+ (Good)</SelectItem>
                    <SelectItem value="5">5+ (Average)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              {/* Clear Filters */}
              <div className="flex items-end">
                <Button 
                  variant="outline" 
                  onClick={clearFilters}
                  className="w-full"
                >
                  Clear All Filters
                </Button>
              </div>
            </div>
          </div>
        )}
        
        {/* Active Filters Display */}
        {activeFiltersCount > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {currentFilters.search && (
              <Badge variant="secondary">
                Search: "{currentFilters.search}"
              </Badge>
            )}
            {currentFilters.niche && (
              <Badge variant="secondary">
                Niche: {currentFilters.niche}
              </Badge>
            )}
            {currentFilters.minScore && (
              <Badge variant="secondary">
                Min Score: {currentFilters.minScore}+
              </Badge>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

