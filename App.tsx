import React from 'react';
import 'react-native-gesture-handler';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { StatusBar } from 'expo-status-bar';
import { InterviewProvider } from './src/state/InterviewContext';
import { colors } from './src/theme';
import { HomeScreen } from './src/screens/HomeScreen';
import { InterviewScreen } from './src/screens/InterviewScreen';
import { FeedbackScreen } from './src/screens/FeedbackScreen';

export type RootStackParamList = {
  Home: undefined;
  Interview: undefined;
  Feedback: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <SafeAreaProvider>
      <GestureHandlerRootView style={{ flex: 1 }}>
        <LinearGradient colors={[colors.backgroundTop, colors.backgroundBottom]} style={{ flex: 1 }}>
          <StatusBar style="light" />
          <InterviewProvider>
            <NavigationContainer>
              <Stack.Navigator
                initialRouteName="Home"
                screenOptions={{
                  headerShown: false,
                  contentStyle: { backgroundColor: 'transparent' },
                }}
              >
                <Stack.Screen name="Home" component={HomeScreen} />
                <Stack.Screen name="Interview" component={InterviewScreen} />
                <Stack.Screen name="Feedback" component={FeedbackScreen} />
              </Stack.Navigator>
            </NavigationContainer>
          </InterviewProvider>
        </LinearGradient>
      </GestureHandlerRootView>
    </SafeAreaProvider>
  );
}
