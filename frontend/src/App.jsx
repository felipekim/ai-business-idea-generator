import { useState, useEffect } from 'react'
import Header from './components/Header.jsx'
import StatsBar from './components/StatsBar.jsx'
import FilterBar from './components/FilterBar.jsx'
import IdeaCard from './components/IdeaCard.jsx'
import LoadingSpinner from './components/LoadingSpinner.jsx'
import { useIdeas, useStats, useNiches, useApi } from './hooks/useApi.js'
import './App.css'

function App() {
  const { apiCall } = useApi()
  const { stats } = useStats()
  const { niches } = useNiches()
  const { ideas, loading, error, refetch } = useIdeas()
  
  const [filters, setFilters] = useState({
    search: '',
    niche: '',
    minScore: '',
    sortBy: 'created_at',
    order: 'desc'
  })
  
  const handleSearch = (searchTerm) => {
    const newFilters = { ...filters, search: searchTerm }
    setFilters(newFilters)
    refetch(newFilters)
  }
  
  const handleNicheFilter = (niche) => {
    const newFilters = { ...filters, niche }
    setFilters(newFilters)
    refetch(newFilters)
  }
  
  const handleScoreFilter = (minScore) => {
    const newFilters = { ...filters, minScore }
    setFilters(newFilters)
    refetch(newFilters)
  }
  
  const handleSort = (sortBy, order) => {
    const newFilters = { ...filters, sortBy, order }
    setFilters(newFilters)
    refetch(newFilters)
  }
  
  const handleRequestValidation = async (ideaId) => {
    try {
      await apiCall(`/ideas/${ideaId}/validate`, {
        method: 'POST',
        body: JSON.stringify({ email: '' })
      })
      
      alert("Validation request submitted! You'll receive results within 24 hours.")
    } catch (err) {
      alert("Failed to submit validation request. Please try again.")
    }
  }
  
  const generateNewIdeas = async () => {
    try {
      await apiCall('/ideas/generate', {
        method: 'POST',
        body: JSON.stringify({ count: 5 })
      })
      
      alert("5 new AI business ideas have been generated!")
      refetch()
    } catch (err) {
      alert("Failed to generate new ideas. Please try again.")
    }
  }
  
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Unable to Load Ideas
            </h2>
            <p className="text-gray-600 mb-4">
              There was an error connecting to the backend service.
            </p>
            <p className="text-sm text-gray-500">
              Error: {error}
            </p>
          </div>
        </div>
      </div>
    )
  }
  
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <StatsBar stats={stats} />
      <FilterBar
        onSearch={handleSearch}
        onNicheFilter={handleNicheFilter}
        onScoreFilter={handleScoreFilter}
        onSort={handleSort}
        niches={niches}
        currentFilters={filters}
      />
      
      <main className="container mx-auto px-4 py-8">
        {/* Generate Ideas Button */}
        <div className="mb-6 text-center">
          <button
            onClick={generateNewIdeas}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors"
          >
            Generate 5 New Ideas
          </button>
          <p className="text-sm text-gray-500 mt-2">
            Click to generate fresh AI business ideas
          </p>
        </div>
        
        {loading ? (
          <LoadingSpinner text="Loading business ideas..." />
        ) : ideas.length === 0 ? (
          <div className="text-center py-12">
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              No Ideas Found
            </h3>
            <p className="text-gray-600 mb-4">
              Try adjusting your filters or generate some new ideas.
            </p>
            <button
              onClick={generateNewIdeas}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
            >
              Generate Ideas
            </button>
          </div>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            {ideas.map((idea) => (
              <IdeaCard
                key={idea.id}
                idea={idea}
                onRequestValidation={handleRequestValidation}
              />
            ))}
          </div>
        )}
        
        {/* Results Info */}
        {ideas.length > 0 && (
          <div className="mt-8 text-center text-sm text-gray-500">
            Showing {ideas.length} business ideas
          </div>
        )}
      </main>
    </div>
  )
}

export default App
