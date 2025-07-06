import { useState } from 'react'
import { 
  Star, 
  DollarSign, 
  Users, 
  Zap, 
  Target, 
  TrendingUp, 
  ChevronDown, 
  ChevronUp,
  ExternalLink,
  CheckCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Progress } from '@/components/ui/progress.jsx'

export default function IdeaCard({ idea, onRequestValidation }) {
  const [isExpanded, setIsExpanded] = useState(false)
  
  const scoreColors = {
    cost_to_build: 'bg-green-500',
    ease_of_implementation: 'bg-blue-500',
    market_size: 'bg-purple-500',
    competition_level: 'bg-orange-500',
    problem_severity: 'bg-red-500',
    founder_fit: 'bg-indigo-500'
  }
  
  const scoreLabels = {
    cost_to_build: 'Cost to Build',
    ease_of_implementation: 'Implementation',
    market_size: 'Market Size',
    competition_level: 'Competition',
    problem_severity: 'Problem Severity',
    founder_fit: 'Founder Fit'
  }
  
  const formatCurrency = (amount) => {
    if (amount >= 1000000) {
      return `$${(amount / 1000000).toFixed(1)}M`
    } else if (amount >= 1000) {
      return `$${(amount / 1000).toFixed(0)}K`
    } else {
      return `$${amount}`
    }
  }
  
  const getScoreColor = (score) => {
    if (score >= 8) return 'text-green-600'
    if (score >= 6) return 'text-yellow-600'
    return 'text-red-600'
  }
  
  return (
    <Card className="hover:shadow-lg transition-shadow duration-300 border-l-4 border-l-blue-500">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-xl font-bold text-gray-900 mb-2">
              {idea.name}
            </CardTitle>
            <p className="text-gray-600 text-sm mb-3">{idea.summary}</p>
            
            <div className="flex items-center space-x-4 mb-3">
              <div className="flex items-center space-x-1">
                <Star className="h-4 w-4 text-yellow-500 fill-current" />
                <span className={`font-bold ${getScoreColor(idea.scores?.total || 0)}`}>
                  {(idea.scores?.total || 0).toFixed(1)}
                </span>
                <span className="text-gray-500 text-sm">/10</span>
              </div>
              
              <Badge variant="secondary" className="text-xs">
                {idea.niche}
              </Badge>
              
              <div className="flex items-center space-x-1 text-green-600">
                <DollarSign className="h-4 w-4" />
                <span className="font-semibold text-sm">
                  {formatCurrency(idea.launch_cost)}
                </span>
              </div>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-2xl font-bold text-blue-600">
              {(idea.scores?.total || 0).toFixed(1)}
            </div>
            <div className="text-xs text-gray-500">Total Score</div>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="pt-0">
        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div className="text-center p-2 bg-gray-50 rounded">
            <div className="text-sm font-semibold text-gray-900">
              {formatCurrency(idea.revenue_1_year)}
            </div>
            <div className="text-xs text-gray-500">1Y Revenue</div>
          </div>
          <div className="text-center p-2 bg-gray-50 rounded">
            <div className="text-sm font-semibold text-gray-900">
              {formatCurrency(idea.revenue_5_year)}
            </div>
            <div className="text-xs text-gray-500">5Y Revenue</div>
          </div>
          <div className="text-center p-2 bg-gray-50 rounded">
            <div className="text-sm font-semibold text-gray-900">
              {idea.target_audience?.split(' ').slice(0, 2).join(' ')}
            </div>
            <div className="text-xs text-gray-500">Target</div>
          </div>
          <div className="text-center p-2 bg-gray-50 rounded">
            <div className="text-sm font-semibold text-gray-900">AI-Powered</div>
            <div className="text-xs text-gray-500">Solution</div>
          </div>
        </div>
        
        {/* Score Breakdown */}
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Score Breakdown</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {Object.entries(idea.scores || {}).map(([key, value]) => {
              if (key === 'total') return null
              return (
                <div key={key} className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${scoreColors[key]}`}></div>
                  <span className="text-xs text-gray-600 flex-1">
                    {scoreLabels[key]}
                  </span>
                  <span className={`text-xs font-semibold ${getScoreColor(value)}`}>
                    {value?.toFixed(1)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
        
        {/* Expandable Details */}
        <div className="border-t pt-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full justify-between text-gray-600 hover:text-gray-900"
          >
            <span>View Details</span>
            {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </Button>
          
          {isExpanded && (
            <div className="mt-4 space-y-4 text-sm">
              <div>
                <h5 className="font-semibold text-gray-700 mb-1">Problem Solved</h5>
                <p className="text-gray-600">{idea.problem_solved}</p>
              </div>
              
              <div>
                <h5 className="font-semibold text-gray-700 mb-1">AI Solution</h5>
                <p className="text-gray-600">{idea.ai_solution}</p>
              </div>
              
              <div>
                <h5 className="font-semibold text-gray-700 mb-1">Implementation</h5>
                <p className="text-gray-600">{idea.implementation}</p>
              </div>
              
              <div>
                <h5 className="font-semibold text-gray-700 mb-1">Revenue Model</h5>
                <p className="text-gray-600">{idea.revenue_model}</p>
              </div>
              
              <div>
                <h5 className="font-semibold text-gray-700 mb-1">Target Audience</h5>
                <p className="text-gray-600">{idea.target_audience}</p>
              </div>
              
              {/* Action Buttons */}
              <div className="flex space-x-2 pt-2">
                <Button 
                  size="sm" 
                  onClick={() => onRequestValidation(idea.id)}
                  className="flex-1"
                >
                  <CheckCircle className="h-4 w-4 mr-1" />
                  Request Deeper Validation
                </Button>
                <Button variant="outline" size="sm">
                  <ExternalLink className="h-4 w-4 mr-1" />
                  Share
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

