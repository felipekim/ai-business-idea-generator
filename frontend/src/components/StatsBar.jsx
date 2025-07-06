import { TrendingUp, Lightbulb, CheckCircle, Users } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card.jsx'

export default function StatsBar({ stats = {} }) {
  const {
    total_ideas = 0,
    total_validations = 0,
    average_score = 0,
    recent_ideas = 0
  } = stats
  
  const statItems = [
    {
      icon: Lightbulb,
      label: 'Total Ideas',
      value: total_ideas.toLocaleString(),
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      icon: CheckCircle,
      label: 'Validations',
      value: total_validations.toLocaleString(),
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    {
      icon: TrendingUp,
      label: 'Avg Score',
      value: average_score.toFixed(1),
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    },
    {
      icon: Users,
      label: 'This Week',
      value: recent_ideas.toLocaleString(),
      color: 'text-orange-600',
      bgColor: 'bg-orange-50'
    }
  ]
  
  return (
    <div className="bg-gray-50 border-b border-gray-200">
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {statItems.map((item, index) => {
            const Icon = item.icon
            return (
              <Card key={index} className="border-0 shadow-sm">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${item.bgColor}`}>
                      <Icon className={`h-5 w-5 ${item.color}`} />
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-gray-900">
                        {item.value}
                      </div>
                      <div className="text-sm text-gray-500">
                        {item.label}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>
    </div>
  )
}

