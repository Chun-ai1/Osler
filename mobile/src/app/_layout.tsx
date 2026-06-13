import {
  Outfit_600SemiBold,
  Outfit_700Bold,
} from '@expo-google-fonts/outfit';
import {
  Sora_400Regular,
  Sora_500Medium,
  Sora_600SemiBold,
} from '@expo-google-fonts/sora';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useFonts } from 'expo-font';
import { Tabs } from 'expo-router';
import { ActivityIndicator, View } from 'react-native';

import { AppProvider } from '@/state/AppContext';
import { colors, fonts } from '@/theme/tokens';

export default function RootLayout() {
  const [loaded] = useFonts({
    Outfit_600SemiBold,
    Outfit_700Bold,
    Sora_400Regular,
    Sora_500Medium,
    Sora_600SemiBold,
  });

  if (!loaded) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.bg }}>
        <ActivityIndicator color={colors.accent} />
      </View>
    );
  }

  return (
    <AppProvider>
      <Tabs
        screenOptions={{
          headerStyle: { backgroundColor: colors.bgCard },
          headerTitleStyle: { fontFamily: fonts.heading, color: colors.accent, fontSize: 18 },
          headerShadowVisible: false,
          tabBarActiveTintColor: colors.accent,
          tabBarInactiveTintColor: colors.textMuted,
          tabBarStyle: { backgroundColor: colors.bgCard, borderTopColor: colors.borderSolid },
          tabBarLabelStyle: { fontFamily: fonts.bodyMedium, fontSize: 11 },
          sceneStyle: { backgroundColor: colors.bg },
        }}>
        <Tabs.Screen
          name="index"
          options={{
            title: 'Patients',
            tabBarIcon: ({ color, size }) => <Ionicons name="people-outline" color={color} size={size} />,
          }}
        />
        <Tabs.Screen
          name="analyze"
          options={{
            title: 'Analyze',
            tabBarIcon: ({ color, size }) => <Ionicons name="medkit-outline" color={color} size={size} />,
          }}
        />
        <Tabs.Screen
          name="reasoning"
          options={{
            title: 'Reasoning',
            tabBarIcon: ({ color, size }) => <Ionicons name="git-network-outline" color={color} size={size} />,
          }}
        />
        <Tabs.Screen
          name="chat"
          options={{
            title: 'Chat',
            tabBarIcon: ({ color, size }) => <Ionicons name="chatbubble-ellipses-outline" color={color} size={size} />,
          }}
        />
        <Tabs.Screen
          name="settings"
          options={{
            title: 'Settings',
            tabBarIcon: ({ color, size }) => <Ionicons name="settings-outline" color={color} size={size} />,
          }}
        />
      </Tabs>
    </AppProvider>
  );
}
